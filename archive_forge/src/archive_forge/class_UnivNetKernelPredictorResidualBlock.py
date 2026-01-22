from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import ModelOutput, PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_univnet import UnivNetConfig
class UnivNetKernelPredictorResidualBlock(nn.Module):
    """
    Implementation of the residual block for the kernel predictor network inside each location variable convolution
    block (LVCBlock).

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
    """

    def __init__(self, config: UnivNetConfig):
        super().__init__()
        self.channels = config.model_in_channels
        self.kernel_size = config.kernel_predictor_conv_size
        self.dropout_prob = config.kernel_predictor_dropout
        self.leaky_relu_slope = config.leaky_relu_slope
        padding = (self.kernel_size - 1) // 2
        self.dropout = nn.Dropout(self.dropout_prob)
        self.conv1 = nn.Conv1d(self.channels, self.channels, self.kernel_size, padding=padding, bias=True)
        self.conv2 = nn.Conv1d(self.channels, self.channels, self.kernel_size, padding=padding, bias=True)

    def forward(self, hidden_states: torch.FloatTensor):
        residual = hidden_states
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        return hidden_states + residual

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv1)
        nn.utils.weight_norm(self.conv2)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        nn.utils.remove_weight_norm(self.conv2)