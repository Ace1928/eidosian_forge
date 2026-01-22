import argparse
import os
import requests
import torch
from PIL import Image
from transformers import SuperPointConfig, SuperPointForKeypointDetection, SuperPointImageProcessor

    Copy/paste/tweak model's weights to our SuperPoint structure.
    