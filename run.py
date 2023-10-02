import torch
import clip
from PIL import Image
from glob import glob
from argparse import ArgumentParser
from pathlib import Path
import shutil
import os


def configuration_parser(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--model_arch', type=str, choices=["RN50", "RN101", "RN50x4",
                                                           "RN50x16", "RN50x64", "ViT-B/32",
                                                           "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
                        default="ViT-B/32")
    parser.add_argument('--input_dir', type=Path, help='image directory', required=True)
    parser.add_argument('--output_dir', type=Path, help='should be empty', required=True)
    parser.add_argument('--labels', type=str, help='coma seperated labels', required=True)
    return parser


def main():
    parser = ArgumentParser()
    parser = configuration_parser(parser)
    args, unknown = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)
    labels = args.labels.split(',')
    model, preprocess = clip.load(args.model_arch, device=device)
    text = clip.tokenize(labels).to(device)

    image_list = glob(str(args.input_dir.joinpath('*')))
    print(image_list)

    os.makedirs(str(args.output_dir), exist_ok=True)
    
    for label_i in labels:
        os.makedirs(str(args.output_dir.joinpath(label_i)))
        
    for img_i in image_list:
        try:
            image = Image.open(img_i)
        except Exception as e:
            print(f'{e} --> cant read image: {img_i}')
            continue

        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            out_label = labels[probs.argmax()]
            
        shutil.copy(img_i, str(args.output_dir.joinpath(out_label)))
        

if __name__ == "__main__":
    main()