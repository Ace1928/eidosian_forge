import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitdet import VitDetConfig
def add_decomposed_relative_positions(attn, queries, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings as introduced in
    [MViT2](https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py).

    Args:
        attn (`torch.Tensor`):
            Attention map.
        queries (`torch.Tensor`):
            Query q in the attention layer with shape (batch_size, queries_height * queries_width, num_channels).
        rel_pos_h (`torch.Tensor`):
            Relative position embeddings (Lh, num_channels) for height axis.
        rel_pos_w (`torch.Tensor`):
            Relative position embeddings (Lw, num_channels) for width axis.
        q_size (`Tuple[int]`):
            Spatial sequence size of query q with (queries_height, queries_width).
        k_size (`Tuple[int]`]):
            Spatial sequence size of key k with (keys_height, keys_width).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    queries_height, queries_width = q_size
    keys_height, keys_width = k_size
    relative_height = get_rel_pos(queries_height, keys_height, rel_pos_h)
    relative_width = get_rel_pos(queries_width, keys_width, rel_pos_w)
    batch_size, _, dim = queries.shape
    r_q = queries.reshape(batch_size, queries_height, queries_width, dim)
    relative_height = torch.einsum('bhwc,hkc->bhwk', r_q, relative_height)
    relative_weight = torch.einsum('bhwc,wkc->bhwk', r_q, relative_width)
    attn = (attn.view(batch_size, queries_height, queries_width, keys_height, keys_width) + relative_height[:, :, :, :, None] + relative_weight[:, :, :, None, :]).view(batch_size, queries_height * queries_width, keys_height * keys_width)
    return attn