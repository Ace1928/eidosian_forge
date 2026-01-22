import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_pvt_v2 import PvtV2Config
class PvtV2SelfAttention(nn.Module):
    """Efficient self-attention mechanism."""

    def __init__(self, config: PvtV2Config, hidden_size: int, num_attention_heads: int, spatial_reduction_ratio: int):
        super().__init__()
        self.linear_attention = config.linear_attention
        self.pruned_heads = set()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({self.hidden_size}) is not a multiple of the number of attention heads ({self.num_attention_heads})')
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)
        self.spatial_reduction_ratio = spatial_reduction_ratio
        if self.linear_attention:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.spatial_reduction = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1)
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
            self.act = nn.GELU()
        elif spatial_reduction_ratio > 1:
            self.spatial_reduction = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=spatial_reduction_ratio, stride=spatial_reduction_ratio)
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, hidden_states) -> torch.Tensor:
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool=False) -> Tuple[torch.Tensor]:
        batch_size, seq_len, num_channels = hidden_states.shape
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        if self.linear_attention:
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            hidden_states = self.spatial_reduction(self.pool(hidden_states)).reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.act(self.layer_norm(hidden_states))
        elif self.spatial_reduction_ratio > 1:
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            hidden_states = self.spatial_reduction(hidden_states).reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_drop(attention_probs)
        context_layer = (attention_probs @ value_layer).transpose(1, 2).reshape(batch_size, seq_len, num_channels)
        context_layer = self.proj(context_layer)
        context_layer = self.proj_drop(context_layer)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads)
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.proj = prune_linear_layer(self.proj, index, dim=1)
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)