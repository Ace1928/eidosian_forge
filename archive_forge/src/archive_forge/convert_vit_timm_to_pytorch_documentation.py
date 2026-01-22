import argparse
from pathlib import Path
import requests
import timm
import torch
from PIL import Image
from timm.data import ImageNetInfo, infer_imagenet_subset
from transformers import DeiTImageProcessor, ViTConfig, ViTForImageClassification, ViTImageProcessor, ViTModel
from transformers.utils import logging

    Copy/paste/tweak model's weights to our ViT structure.
    