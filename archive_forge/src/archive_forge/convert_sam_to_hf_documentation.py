import argparse
import re
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (

Convert SAM checkpoints from the original repository.

URL: https://github.com/facebookresearch/segment-anything.

Also supports converting the SlimSAM checkpoints from https://github.com/czg1225/SlimSAM/tree/master.
