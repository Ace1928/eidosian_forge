from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_gpt_neox_japanese import GPTNeoXJapaneseConfig
class GPTNeoXJapaneseAttention(nn.Module):

    def __init__(self, config, use_bias=False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base)
        self.max_positions = config.max_position_embeddings
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.use_bias = use_bias
        self.dense_bias = nn.Parameter(torch.zeros(config.hidden_size)) if use_bias else None

    def forward(self, hidden_states, attention_mask, head_mask=None, layer_past=None, use_cache=False, output_attentions=False):
        has_layer_past = layer_past is not None and layer_past[0].numel() > 0
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
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
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
        return (outputs, self.dense_bias)

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

    def _create_causal_mask(self, key_length, query_length):
        causal_mask = torch.tril(torch.ones((self.max_positions, self.max_positions), dtype=torch.bool).view(1, 1, self.max_positions, self.max_positions))
        return causal_mask[:, :, key_length - query_length:key_length, :key_length]

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)
        causal_mask = self._create_causal_mask(key_length, query_length)
        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(batch_size * num_attention_heads, query_length, key_length, dtype=query.dtype, device=key.device)
        attn_scores = torch.baddbmm(attn_scores, query, key.transpose(1, 2), beta=1.0, alpha=torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor)
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)
        mask_value = torch.finfo(attn_scores.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        causal_mask = causal_mask.to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)