import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
class JukeboxConv1D(nn.Module):

    def __init__(self, input_width, output_width):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        weight = torch.empty(input_width, output_width)
        bias = torch.zeros(output_width)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, hidden_states):
        size_out = (*hidden_states.size()[:-1], self.output_width)
        hidden_states = torch.addmm(self.bias.type_as(hidden_states), hidden_states.view(-1, hidden_states.size(-1)), self.weight.type_as(hidden_states))
        hidden_states = hidden_states.view(*size_out)
        return hidden_states