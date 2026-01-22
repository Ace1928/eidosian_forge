import argparse
import json
from pathlib import Path
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, Swinv2Config, Swinv2ForImageClassification
Convert Swinv2 checkpoints from the timm library.