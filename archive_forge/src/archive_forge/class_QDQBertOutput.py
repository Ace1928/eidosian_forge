import math
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_qdqbert import QDQBertConfig
class QDQBertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = quant_nn.QuantLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.add_local_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.add_residual_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        add_local = self.add_local_input_quantizer(hidden_states)
        add_residual = self.add_residual_input_quantizer(input_tensor)
        hidden_states = self.LayerNorm(add_local + add_residual)
        return hidden_states