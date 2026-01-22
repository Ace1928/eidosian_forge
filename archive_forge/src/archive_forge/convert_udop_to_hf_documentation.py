import argparse
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T
from transformers import (
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
Convert UDOP checkpoints from the original repository. URL: https://github.com/microsoft/i-Code/tree/main/i-Code-Doc