import argparse
import re
import requests
import torch
from models.blip import blip_decoder
from models.blip_itm import blip_itm
from models.blip_vqa import blip_vqa
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import (

    Copy/paste/tweak model's weights to transformers design.
    