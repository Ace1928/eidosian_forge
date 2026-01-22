import argparse
import collections
import json
from pathlib import Path
import requests
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging

    Copy/paste/tweak model's weights to our MobileViTV2 structure.
    