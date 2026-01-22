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
class CpmAntEncoder(nn.Module):

    def __init__(self, config: CpmAntConfig):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.layers = nn.ModuleList([CpmAntTransformerBlock(config) for ith in range(self.num_layers)])
        self.output_layernorm = CpmAntLayerNorm(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_bias: torch.Tensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache: Optional[bool]=None):
        """
        Args:
            hidden_states (`torch.Tensor`):
                Input to the layer of shape `(batch, seq_len, dim_model)`
            attention_mask (`torch.Tensor`):
                Avoid invalid areas to participate in the calculation of shape `(batch, seq_len, seq_len)`
            position_bias (`torch.Tensor`):
                Provides position information to attention mechanism of shape `(num_heads, seq_len, seq_len)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            past_key_values (`Tuple[torch.Tensor, torch.Tensor])`, *optional*):
                Cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        current_key_values = () if use_cache else None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(hidden_states, attention_mask, position_bias, output_attentions=output_attentions, past_key_values=past_key_values[i] if past_key_values else None, use_cache=use_cache)
            hidden_states, attn_weights, current_key_value = layer_outputs
            if output_attentions:
                all_self_attns += (attn_weights,)
            if current_key_value is not None:
                current_key_values = current_key_values + (current_key_value,)
        hidden_states = self.output_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return (hidden_states, current_key_values, all_hidden_states, all_self_attns)