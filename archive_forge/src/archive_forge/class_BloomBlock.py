import math
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_bloom import BloomConfig
class BloomBlock(nn.Module):

    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = BloomMLP(config)
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
        layernorm_output = self.input_layernorm(hidden_states)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        attn_outputs = self.self_attention(layernorm_output, residual, layer_past=layer_past, attention_mask=attention_mask, alibi=alibi, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        layernorm_output = self.post_attention_layernorm(attention_output)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output
        output = self.mlp(layernorm_output, residual)
        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]
        return outputs