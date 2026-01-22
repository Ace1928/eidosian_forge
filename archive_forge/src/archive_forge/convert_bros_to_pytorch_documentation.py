import argparse
import bros  # original repo
import torch
from transformers import BrosConfig, BrosModel, BrosProcessor
from transformers.utils import logging
Convert Bros checkpoints.