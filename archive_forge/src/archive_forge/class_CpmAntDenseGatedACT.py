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
class CpmAntDenseGatedACT(nn.Module):

    def __init__(self, config: CpmAntConfig):
        super().__init__()
        self.w_0 = nn.Linear(config.hidden_size, config.dim_ff, bias=False)
        self.w_1 = nn.Linear(config.hidden_size, config.dim_ff, bias=False)
        self.act = torch.nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        """Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        gate_score = self.act(self.w_0(hidden_states))
        hidden_states = self.w_1(hidden_states)
        hidden_states = gate_score * hidden_states
        return hidden_states