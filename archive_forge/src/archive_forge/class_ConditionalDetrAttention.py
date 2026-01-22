import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_conditional_detr import ConditionalDetrConfig
class ConditionalDetrAttention(nn.Module):
    """
    Cross-Attention used in Conditional DETR 'Conditional DETR for Fast Training Convergence' paper.

    The key q_proj, k_proj, v_proj are defined outside the attention. This attention allows the dim of q, k to be
    different to v.
    """

    def __init__(self, embed_dim: int, out_dim: int, num_heads: int, dropout: float=0.0, bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.v_head_dim = out_dim // num_heads
        if self.v_head_dim * num_heads != self.out_dim:
            raise ValueError(f'out_dim must be divisible by num_heads (got `out_dim`: {self.out_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=bias)

    def _qk_shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _v_shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, key_states: Optional[torch.Tensor]=None, value_states: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        batch_size, target_len, _ = hidden_states.size()
        query_states = hidden_states * self.scaling
        key_states = self._qk_shape(key_states, -1, batch_size)
        value_states = self._v_shape(value_states, -1, batch_size)
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        v_proj_shape = (batch_size * self.num_heads, -1, self.v_head_dim)
        query_states = self._qk_shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*v_proj_shape)
        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(f'Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(f'Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (batch_size * self.num_heads, target_len, self.v_head_dim):
            raise ValueError(f'`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.v_head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.v_head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, self.out_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped)