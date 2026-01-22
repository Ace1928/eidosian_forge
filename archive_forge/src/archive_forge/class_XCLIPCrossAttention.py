from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_x_clip import XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig
class XCLIPCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.prompt_num_attention_heads
        dim = config.projection_dim
        head_dim = dim // self.num_heads
        self.scale = head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(config.prompt_attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(config.prompt_projection_dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, queries, keys, values):
        """Input shape: Batch x Time x Channel"""
        batch_size, query_seq_len, hidden_size = queries.shape
        batch_size, key_seq_len, hidden_size = keys.shape
        queries = self.q_proj(queries).reshape(batch_size, query_seq_len, self.num_heads, hidden_size // self.num_heads).permute(0, 2, 1, 3)
        keys = self.k_proj(keys).reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads).permute(0, 2, 1, 3)
        values = self.v_proj(values).reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads).permute(0, 2, 1, 3)
        attn = queries @ keys.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ values).transpose(1, 2).reshape(batch_size, query_seq_len, hidden_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x