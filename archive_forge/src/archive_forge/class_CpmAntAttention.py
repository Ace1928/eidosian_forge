import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig
class CpmAntAttention(nn.Module):

    def __init__(self, config: CpmAntConfig):
        super().__init__()
        self.dim_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dim_head = config.dim_head
        self.project_q = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        self.project_k = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        self.project_v = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        self.attention_out = nn.Linear(self.num_heads * self.dim_head, self.dim_model, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        if config.dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=config.dropout_p)
        else:
            self.dropout = None

    def forward(self, hidden_q: torch.Tensor, hidden_kv: torch.Tensor, attention_mask: torch.BoolTensor, position_bias: torch.Tensor, output_attentions: Optional[bool]=False, past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache: Optional[bool]=None):
        """
        Args:
            hidden_q (`torch.Tensor`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            hidden_kv (`torch.Tensor` of shape `(batch, len_k, dim_model)`)):
                Tensor *key_value* and *query* of shape `(batch, len_k, dim_model)`
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            past_key_values (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
                Cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(1)
        len_k = hidden_kv.size(1)
        query = self.project_q(hidden_q)
        key = self.project_k(hidden_kv)
        value = self.project_v(hidden_kv)
        query = query.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        if past_key_values is not None:
            key = torch.cat([past_key_values[0], key], dim=-2)
            value = torch.cat([past_key_values[1], value], dim=-2)
            len_k = key.size(-2)
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim_head)
        score = score + position_bias
        score = torch.masked_fill(score, attention_mask.view(batch_size, 1, len_q, len_k) == torch.tensor(False), torch.scalar_tensor(float('-inf'), device=score.device, dtype=score.dtype))
        score = self.softmax(score)
        score = torch.masked_fill(score, attention_mask.view(batch_size, 1, len_q, len_k) == torch.tensor(False), torch.scalar_tensor(0, device=score.device, dtype=score.dtype))
        if output_attentions:
            attn_weights = score
        else:
            attn_weights = None
        if self.dropout is not None:
            score = self.dropout(score)
        score = torch.matmul(score, value)
        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3)
        score = score.contiguous().view(batch_size, len_q, self.num_heads * self.dim_head)
        score = self.attention_out(score)
        past_key_values = None
        if use_cache:
            past_key_values = (key, value)
        return (score, attn_weights, past_key_values)