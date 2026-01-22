import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
class BarkBlock(nn.Module):

    def __init__(self, config, is_causal=False):
        super().__init__()
        if is_causal:
            self.layernorm_1 = BarkLayerNorm(config.hidden_size, bias=config.bias)
            self.layernorm_2 = BarkLayerNorm(config.hidden_size, bias=config.bias)
        else:
            self.layernorm_1 = nn.LayerNorm(config.hidden_size)
            self.layernorm_2 = nn.LayerNorm(config.hidden_size)
        self.attn = BARK_ATTENTION_CLASSES[config._attn_implementation](config, is_causal=is_causal)
        self.mlp = BarkMLP(config)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        intermediary_hidden_states = self.layernorm_1(hidden_states)
        attn_outputs = self.attn(intermediary_hidden_states, past_key_values=past_key_values, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        intermediary_hidden_states = hidden_states + attn_output
        intermediary_hidden_states = intermediary_hidden_states + self.mlp(self.layernorm_2(intermediary_hidden_states))
        if use_cache:
            outputs = (intermediary_hidden_states,) + outputs
        else:
            outputs = (intermediary_hidden_states,) + outputs[1:]
        return outputs