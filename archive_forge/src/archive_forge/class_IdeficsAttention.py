from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_idefics import IdeficsConfig
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer
class IdeficsAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float=0.0, is_cross_attention: bool=False, config: PretrainedConfig=None, qk_layer_norms: bool=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.is_causal = True
        if self.head_dim * num_heads != self.hidden_size:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {num_heads}).')
        self.is_cross_attention = is_cross_attention
        if not hasattr(nn.functional, 'scaled_dot_product_attention'):
            raise ValueError('this model requires pytorch 2.0 or higher')
        if self.is_cross_attention:
            kv_input_dim = self.hidden_size if not hasattr(config.vision_config, 'embed_dim') else config.vision_config.embed_dim
            self.q_proj = nn.Linear(self.hidden_size, num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(kv_input_dim, num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(kv_input_dim, num_heads * self.head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = IdeficsEmbedding(self.head_dim)
        self.qk_layer_norms = qk_layer_norms
        if self.qk_layer_norms:
            self.q_layer_norm = IdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layer_norm = IdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: bool=False, use_cache: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        is_cross_attention = self.is_cross_attention or key_value_states is not None
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if not is_cross_attention:
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            _, kv_len, _ = key_value_states.size()
            key_states = self.k_proj(key_value_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(key_value_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        if not is_cross_attention:
            cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, q_len))
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
        if self.qk_layer_norms:
            query_states = self.q_layer_norm(query_states)
            key_states = self.k_layer_norm(key_states)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}')
        if query_states.device.type == 'cuda' and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=self.dropout, is_causal=self.is_causal and attention_mask is None and (q_len > 1))
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_weights = None
        if output_attentions:
            logger.warning_once('attn_weights are not extracted in scaled_dot_product_attention. The model returns None instead')
        return (attn_output, attn_weights, past_key_value)