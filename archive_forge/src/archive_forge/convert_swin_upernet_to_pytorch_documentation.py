import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import SegformerImageProcessor, SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
Convert Swin Transformer + UperNet checkpoints from mmsegmentation.

URL: https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin
