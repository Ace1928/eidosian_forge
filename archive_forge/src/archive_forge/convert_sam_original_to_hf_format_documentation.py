import argparse
import re
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (

Convert SAM checkpoints from the original repository.
