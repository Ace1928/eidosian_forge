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
def to_f32(params):
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, params)