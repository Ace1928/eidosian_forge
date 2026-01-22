import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class EfficientFormerSelfAttention(nn.Module):

    def __init__(self, dim: int, key_dim: int, num_heads: int, attention_ratio: int, resolution: int):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.scale = key_dim ** (-0.5)
        self.total_key_dim = key_dim * num_heads
        self.expanded_key_dim = int(attention_ratio * key_dim)
        self.total_expanded_key_dim = int(self.expanded_key_dim * num_heads)
        hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2
        self.qkv = nn.Linear(dim, hidden_size)
        self.projection = nn.Linear(self.total_expanded_key_dim, dim)
        points = list(itertools.product(range(resolution), range(resolution)))
        num_points = len(points)
        attention_offsets = {}
        idxs = []
        for point_1 in points:
            for point_2 in points:
                offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(num_points, num_points))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool=False) -> Tuple[torch.Tensor]:
        batch_size, sequence_length, num_channels = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query_layer, key_layer, value_layer = qkv.reshape(batch_size, sequence_length, self.num_heads, -1).split([self.key_dim, self.key_dim, self.expanded_key_dim], dim=3)
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)
        if not self.training:
            self.ab = self.ab.to(self.attention_biases.device)
        attention_probs = torch.matmul(query_layer, key_layer.transpose(-2, -1)) * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        attention_probs = attention_probs.softmax(dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer).transpose(1, 2)
        context_layer = context_layer.reshape(batch_size, sequence_length, self.total_expanded_key_dim)
        context_layer = self.projection(context_layer)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs