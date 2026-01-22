import argparse
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from transformers import (
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

    Copy/paste/tweak model's weights to Transformers design.
    