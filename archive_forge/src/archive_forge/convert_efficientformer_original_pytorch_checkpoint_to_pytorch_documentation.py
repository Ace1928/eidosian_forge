import argparse
import re
from pathlib import Path
import requests
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import (
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
Convert EfficientFormer checkpoints from the original repository.

URL: https://github.com/snap-research/EfficientFormer
