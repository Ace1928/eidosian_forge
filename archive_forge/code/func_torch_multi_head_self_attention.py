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