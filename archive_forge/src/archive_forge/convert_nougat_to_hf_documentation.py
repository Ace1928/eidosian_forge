import argparse
import torch
from huggingface_hub import hf_hub_download
from nougat import NougatModel
from nougat.dataset.rasterize import rasterize_paper
from nougat.utils.checkpoint import get_checkpoint
from PIL import Image
from transformers import (
Convert Nougat checkpoints using the original `nougat` library. URL:
https://github.com/facebookresearch/nougat/tree/main