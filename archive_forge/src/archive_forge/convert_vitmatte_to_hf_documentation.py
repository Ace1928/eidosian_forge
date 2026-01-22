import argparse
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor
Convert VitMatte checkpoints from the original repository.

URL: https://github.com/hustvl/ViTMatte
