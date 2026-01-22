import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2CLS
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_swiftformer import SwiftFormerConfig
class SwiftFormerEfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int=512):
        super().__init__()
        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)
        self.w_g = nn.Parameter(torch.randn(dim, 1))
        self.scale_factor = dim ** (-0.5)
        self.proj = nn.Linear(dim, dim)
        self.final = nn.Linear(dim, dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)
        query_weight = query @ self.w_g
        scaled_query_weight = query_weight * self.scale_factor
        scaled_query_weight = scaled_query_weight.softmax(dim=-1)
        global_queries = torch.sum(scaled_query_weight * query, dim=1)
        global_queries = global_queries.unsqueeze(1).repeat(1, key.shape[1], 1)
        out = self.proj(global_queries * key) + query
        out = self.final(out)
        return out