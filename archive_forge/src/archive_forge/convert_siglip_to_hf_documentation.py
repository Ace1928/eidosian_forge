import argparse
import collections
from pathlib import Path
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from numpy import load
from PIL import Image
from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipProcessor, SiglipTokenizer
from transformers.utils import logging

    Copy/paste/tweak model's weights to our SigLIP structure.
    