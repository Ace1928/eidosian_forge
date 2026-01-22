import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
from ....utils import logging
from .configuration_mctct import MCTCTConfig
class MCTCTLayerNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.singleton_weight = nn.Parameter(torch.ones(1))
        self.singleton_bias = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states):
        return hidden_states * self.singleton_weight + self.singleton_bias