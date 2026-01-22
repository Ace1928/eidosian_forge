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
class LevitSubsample(nn.Module):

    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, hidden_state):
        batch_size, _, channels = hidden_state.shape
        hidden_state = hidden_state.view(batch_size, self.resolution, self.resolution, channels)[:, ::self.stride, ::self.stride].reshape(batch_size, -1, channels)
        return hidden_state