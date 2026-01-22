import argparse
import collections
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration
from transformers.utils import logging
Loads the config and model, converts the T5X checkpoint, and saves a PyTorch checkpoint.