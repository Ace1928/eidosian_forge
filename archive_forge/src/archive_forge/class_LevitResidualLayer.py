import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig
class LevitResidualLayer(nn.Module):
    """
    Residual Block for LeViT
    """

    def __init__(self, module, drop_rate):
        super().__init__()
        self.module = module
        self.drop_rate = drop_rate

    def forward(self, hidden_state):
        if self.training and self.drop_rate > 0:
            rnd = torch.rand(hidden_state.size(0), 1, 1, device=hidden_state.device)
            rnd = rnd.ge_(self.drop_rate).div(1 - self.drop_rate).detach()
            hidden_state = hidden_state + self.module(hidden_state) * rnd
            return hidden_state
        else:
            hidden_state = hidden_state + self.module(hidden_state)
            return hidden_state