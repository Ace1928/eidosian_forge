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
class LevitClassificationLayer(nn.Module):
    """
    LeViT Classification Layer
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_state):
        hidden_state = self.batch_norm(hidden_state)
        logits = self.linear(hidden_state)
        return logits