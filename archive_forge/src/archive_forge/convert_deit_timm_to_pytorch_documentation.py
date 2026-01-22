import argparse
import json
from pathlib import Path
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DeiTConfig, DeiTForImageClassificationWithTeacher, DeiTImageProcessor
from transformers.utils import logging

    Copy/paste/tweak model's weights to our DeiT structure.
    