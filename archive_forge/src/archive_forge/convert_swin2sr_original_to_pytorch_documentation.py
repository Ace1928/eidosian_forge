import argparse
import requests
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution, Swin2SRImageProcessor
Convert Swin2SR checkpoints from the original repository. URL: https://github.com/mv-lab/swin2sr