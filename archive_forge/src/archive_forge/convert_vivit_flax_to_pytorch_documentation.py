import argparse
import json
import os.path
from collections import OrderedDict
import numpy as np
import requests
import torch
from flax.training.checkpoints import restore_checkpoint
from huggingface_hub import hf_hub_download
from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor
from transformers.image_utils import PILImageResampling
Convert Flax ViViT checkpoints from the original repository to PyTorch. URL:
https://github.com/google-research/scenic/tree/main/scenic/projects/vivit
