import argparse
from collections import OrderedDict
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import GLPNConfig, GLPNForDepthEstimation, GLPNImageProcessor
from transformers.utils import logging

    Copy/paste/tweak model's weights to our GLPN structure.
    