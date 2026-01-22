import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig
class LevitAttention(nn.Module):

    def __init__(self, hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim ** (-0.5)
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads * 2
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads
        self.queries_keys_values = MLPLayerWithBN(hidden_sizes, self.out_dim_keys_values)
        self.activation = nn.Hardswish()
        self.projection = MLPLayerWithBN(self.out_dim_projection, hidden_sizes, bn_weight_init=0)
        points = list(itertools.product(range(resolution), range(resolution)))
        len_points = len(points)
        attention_offsets, indices = ({}, [])
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])
        self.attention_bias_cache = {}
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_attention_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(indices).view(len_points, len_points), persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}

    def get_attention_biases(self, device):
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, hidden_state):
        batch_size, seq_length, _ = hidden_state.shape
        queries_keys_values = self.queries_keys_values(hidden_state)
        query, key, value = queries_keys_values.view(batch_size, seq_length, self.num_attention_heads, -1).split([self.key_dim, self.key_dim, self.attention_ratio * self.key_dim], dim=3)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(hidden_state.device)
        attention = attention.softmax(dim=-1)
        hidden_state = (attention @ value).transpose(1, 2).reshape(batch_size, seq_length, self.out_dim_projection)
        hidden_state = self.projection(self.activation(hidden_state))
        return hidden_state