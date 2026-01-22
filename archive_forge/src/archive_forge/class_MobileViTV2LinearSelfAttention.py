from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilevitv2 import MobileViTV2Config
class MobileViTV2LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in MobileViTV2 paper:
    https://arxiv.org/abs/2206.02680

    Args:
        config (`MobileVitv2Config`):
             Model configuration object
        embed_dim (`int`):
            `input_channels` from an expected input of size :math:`(batch_size, input_channels, height, width)`
    """

    def __init__(self, config: MobileViTV2Config, embed_dim: int) -> None:
        super().__init__()
        self.qkv_proj = MobileViTV2ConvLayer(config=config, in_channels=embed_dim, out_channels=1 + 2 * embed_dim, bias=True, kernel_size=1, use_normalization=False, use_activation=False)
        self.attn_dropout = nn.Dropout(p=config.attn_dropout)
        self.out_proj = MobileViTV2ConvLayer(config=config, in_channels=embed_dim, out_channels=embed_dim, bias=True, kernel_size=1, use_normalization=False, use_activation=False)
        self.embed_dim = embed_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        query, key, value = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)
        context_scores = torch.nn.functional.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        out = torch.nn.functional.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out