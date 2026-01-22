import argparse
import collections
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from clip.model import CLIP
from flax.training import checkpoints
from huggingface_hub import Repository
from transformers import (

    Copy/paste/tweak model's weights to transformers design.
    