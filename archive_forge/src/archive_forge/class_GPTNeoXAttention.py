from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10, logging
from .configuration_gpt_neox import GPTNeoXConfig
class GPTNeoXAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError('The hidden size is not divisble by the number of attention heads! Make sure to update them')
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self._init_bias(config.max_position_embeddings)
        self.register_buffer('masked_bias', torch.tensor(-1000000000.0), persistent=False)
        self._init_rope()
        self.norm_factor = self.head_size ** (-0.5)
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.is_causal = True

    def _init_bias(self, max_positions, device=None):
        self.register_buffer('bias', torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions), persistent=False)
        if device is not None:
            self.bias = self.bias.to(device)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = GPTNeoXRotaryEmbedding(self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base)
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'linear':
                self.rotary_emb = GPTNeoXLinearScalingRotaryEmbedding(self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base, scaling_factor=scaling_factor)
            elif scaling_type == 'dynamic':
                self.rotary_emb = GPTNeoXDynamicNTKScalingRotaryEmbedding(self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base, scaling_factor=scaling_factor)
            else:
                raise ValueError(f'Unknown RoPE scaling type {scaling_type}')

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.FloatTensor, position_ids: torch.LongTensor, head_mask: Optional[torch.FloatTensor]=None, layer_past: Optional[Tuple[torch.Tensor]]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, padding_mask: Optional[torch.Tensor]=None):
        has_layer_past = layer_past is not None
        qkv = self.query_key_value(hidden_states)
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)
        query = qkv[..., :self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size:2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)
        query_rot = query[..., :self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., :self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(batch_size * num_attention_heads, query_length, key_length, dtype=query.dtype, device=key.device)
        attn_scores = torch.baddbmm(attn_scores, query, key.transpose(1, 2), beta=1.0, alpha=self.norm_factor)
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)
        mask_value = torch.finfo(attn_scores.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_weights = self.attention_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)