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