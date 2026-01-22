import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_prophetnet import ProphetNetConfig
class ProphetNetAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ProphetNetConfig, num_attn_heads: int):
        super().__init__()
        hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.dropout = config.dropout
        self.num_attn_heads = num_attn_heads
        self.head_dim = hidden_size // num_attn_heads
        assert self.head_dim * num_attn_heads == hidden_size, '`config.hidden_size` must be divisible by `config.num_encoder_attention_heads` and `config.num_decoder_attention_heads`'
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, key_value_states: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None, layer_head_mask: Optional[Tensor]=None, past_key_value: Optional[Tuple[Tensor]]=None, output_attentions: bool=False) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, tgt_len, hidden_size = hidden_states.size()
        is_cross_attention = key_value_states is not None
        assert list(hidden_states.size()) == [batch_size, tgt_len, hidden_size], f'Size of hidden states should be {(batch_size, tgt_len, hidden_size)}, but is {hidden_states.size()}'
        query_states = self.query_proj(hidden_states) / self.head_dim ** 0.5
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.key_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.value_proj(key_value_states), -1, batch_size)
        else:
            key_states = self._shape(self.key_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.value_proj(hidden_states), -1, batch_size)
        if is_cross_attention:
            past_key_value = (key_states, value_states)
        proj_shape = (batch_size, self.num_attn_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(2)
        attn_weights = torch.einsum('bsij,bsjk->bsik', query_states, key_states.transpose(2, 3))
        expected_shape = (batch_size, self.num_attn_heads, tgt_len, src_len)
        if attn_weights.size() != expected_shape:
            raise ValueError(f'Attention weights should have size {expected_shape}, but is {attn_weights.size()}')
        if attention_mask is not None and attention_mask.dim() == 0:
            attention_mask = None
        expected_shape = (batch_size, self.num_attn_heads, 1, src_len)
        if attention_mask is not None and attention_mask.size() != expected_shape:
            raise ValueError(f'Attention mask should have size {expected_shape}, but is {attention_mask.size()}')
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if output_attentions:
            attn_weights_reshaped = attn_weights
        else:
            attn_weights_reshaped = None
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_attn_heads,), f'Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is {layer_head_mask.size()}'
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(batch_size, self.num_attn_heads, tgt_len, src_len)
            attn_weights_reshaped = layer_head_mask.view(1, -1, 1, 1) * attn_weights_reshaped
        attn_probs = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.einsum('bsij,bsjk->bsik', attn_probs, value_states)
        expected_shape = (batch_size, self.num_attn_heads, tgt_len, self.head_dim)
        if attn_output.size() != expected_shape:
            raise ValueError(f'`attn_output` should have shape {expected_shape}, but is of shape {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, hidden_size)
        attn_output = self.out_proj(attn_output)
        attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)
        return (attn_output, attn_weights_reshaped, past_key_value)