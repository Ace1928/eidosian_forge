from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gemma.configuration_gemma import GemmaConfig
class UnpaddedGemmaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(self, cos_sin: Tuple[torch.Tensor, torch.Tensor], nz_hidden_states: torch.Tensor, nz_position_ids: torch.LongTensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        query_states = self.q_proj(nz_hidden_states).view(-1, self.num_heads, self.head_dim)
        key_states = self.k_proj(nz_hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(nz_hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, nz_position_ids)
        attn_output = flash_attn_varlen_func(q=query_states, k=key_states, v=value_states, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, dropout_p=0.0, causal=True)
        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)