import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
def _build_network_inputs(self, inputs):
    """Construct the final input, including position encoding."""
    batch_size = inputs.shape[0]
    index_dims = inputs.shape[1:-1]
    if self.position_encoding_type == 'trainable':
        pos_enc = self.position_embeddings(batch_size)
    elif self.position_encoding_type == 'fourier':
        pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)
    pos_enc = self.positions_projection(pos_enc)
    if self.concat_or_add_pos == 'concat':
        inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
    elif self.concat_or_add_pos == 'add':
        inputs_with_pos = inputs + pos_enc
    return (inputs_with_pos, inputs)