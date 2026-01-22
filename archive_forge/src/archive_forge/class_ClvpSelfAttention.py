import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (
class ClvpSelfAttention(nn.Module):
    """
    Multi-headed attention to combine Absolute and Rotary Positional Embeddings into a single Attention module.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale = self.head_dim ** (-0.5)
        self.dropout = config.attention_dropout
        if hasattr(config, 'max_position_embeddings'):
            max_positions = config.max_position_embeddings
            bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))
            bias = bias.view(1, 1, max_positions, max_positions)
            self.register_buffer('bias', bias, persistent=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.FloatTensor, rotary_pos_emb: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, use_cache: Optional[bool]=False, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        if rotary_pos_emb is not None and position_ids is None:
            raise ValueError('`position_ids` must be provided when `rotary_pos_emb` is not None.')
        bsz, _, embed_dim = hidden_states.size()
        query_states = self._shape(self.q_proj(hidden_states), -1, bsz) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None
        if rotary_pos_emb is not None:
            rotary_emb_dim = rotary_pos_emb.shape[-1]
            query_rot, query_pass = (query_states[..., :rotary_emb_dim], query_states[..., rotary_emb_dim:])
            key_rot, key_pass = (key_states[..., :rotary_emb_dim], key_states[..., rotary_emb_dim:])
            value_rot, value_pass = (value_states[..., :rotary_emb_dim], value_states[..., rotary_emb_dim:])
            cos, sin = (rotary_pos_emb.cos().squeeze(0), rotary_pos_emb.sin().squeeze(0))
            query_rot, key_rot, value_rot = apply_rotary_pos_emb(query_rot, key_rot, value_rot, cos, sin, position_ids)
            query_states = torch.cat((query_rot, query_pass), dim=-1)
            key_states = torch.cat((key_rot, key_pass), dim=-1)
            value_states = torch.cat((value_rot, value_pass), dim=-1)
        tgt_len = query_states.shape[2]
        src_len = key_states.shape[2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, present, attn_weights)