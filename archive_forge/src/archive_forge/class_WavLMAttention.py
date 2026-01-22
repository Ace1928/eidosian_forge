import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wavlm import WavLMConfig
class WavLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, num_buckets: int=320, max_distance: int=800, has_relative_position_bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)
        if has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_bias: Optional[torch.Tensor]=None, output_attentions: bool=False, index=0) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Attention layer with relative attention"""
        bsz, tgt_len, _ = hidden_states.size()
        if position_bias is None:
            position_bias = self.compute_bias(tgt_len, tgt_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, tgt_len)
        gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        relative_position_proj = relative_position_proj.view(gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
        gated_position_bias = gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
        gated_position_bias = gated_position_bias.view((-1, tgt_len, tgt_len))
        attn_output, attn_weights = self.torch_multi_head_self_attention(hidden_states, attention_mask, gated_position_bias, output_attentions)
        return (attn_output, attn_weights, position_bias)

    def torch_multi_head_self_attention(self, hidden_states: torch.FloatTensor, attention_mask: Union[torch.LongTensor, torch.BoolTensor], gated_position_bias: torch.FloatTensor, output_attentions: bool) -> (torch.FloatTensor, torch.FloatTensor):
        """simple wrapper around torch's multi_head_attention_forward function"""
        query = key = value = hidden_states.transpose(0, 1)
        key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None
        bias_k = bias_v = None
        add_zero_attn = False
        attn_output, attn_weights = F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, torch.empty([0]), torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)), bias_k, bias_v, add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask, output_attentions, gated_position_bias, use_separate_proj_weight=True, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight)
        attn_output = attn_output.transpose(0, 1)
        if attn_weights is not None:
            attn_weights = attn_weights[:, None].broadcast_to(attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:])
        return (attn_output, attn_weights)

    def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor:
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        values = self.rel_attn_embed(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(self, relative_positions: torch.FloatTensor) -> torch.FloatTensor:
        num_buckets = self.num_buckets // 2
        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
        relative_positions_if_large = relative_positions_if_large / math.log(self.max_distance / max_exact)
        relative_positions_if_large = relative_positions_if_large * (num_buckets - max_exact)
        relative_position_if_large = (max_exact + relative_positions_if_large).to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
        return relative_buckets