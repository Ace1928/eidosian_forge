import argparse
import json
import pickle
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, SwinConfig
from transformers.utils import logging

    Copy/paste/tweak model's weights to our MaskFormer structure.
    