import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ibert import IBertConfig
from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear
class IBertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.weight_bit = 8
        self.bias_bit = 32
        self.ln_input_bit = 22
        self.ln_output_bit = 32
        self.dense = QuantLinear(config.hidden_size, config.hidden_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        self.ln_input_act = QuantAct(self.ln_input_bit, quant_mode=self.quant_mode)
        self.LayerNorm = IntLayerNorm(config.hidden_size, eps=config.layer_norm_eps, output_bit=self.ln_output_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, hidden_states_scaling_factor, input_tensor, input_tensor_scaling_factor):
        hidden_states, hidden_states_scaling_factor = self.dense(hidden_states, hidden_states_scaling_factor)
        hidden_states = self.dropout(hidden_states)
        hidden_states, hidden_states_scaling_factor = self.ln_input_act(hidden_states, hidden_states_scaling_factor, identity=input_tensor, identity_scaling_factor=input_tensor_scaling_factor)
        hidden_states, hidden_states_scaling_factor = self.LayerNorm(hidden_states, hidden_states_scaling_factor)
        hidden_states, hidden_states_scaling_factor = self.output_activation(hidden_states, hidden_states_scaling_factor)
        return (hidden_states, hidden_states_scaling_factor)