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
def _convert_attn_layers(params):
    new_params = {}
    processed_attn_layers = []
    for k, v in params.items():
        if 'attn.' in k:
            base = k[:k.rindex('attn.') + 5]
            if base in processed_attn_layers:
                continue
            processed_attn_layers.append(base)
            dim = params[base + 'out.weight'].shape[-1]
            new_params[base + 'out_proj.weight'] = params[base + 'out.weight'].reshape(dim, dim).T
            new_params[base + 'out_proj.bias'] = params[base + 'out.bias']
        else:
            new_params[k] = v
    return new_params