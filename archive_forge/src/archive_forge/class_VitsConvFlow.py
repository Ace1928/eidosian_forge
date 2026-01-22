import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
class VitsConvFlow(nn.Module):

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.filter_channels = config.hidden_size
        self.half_channels = config.depth_separable_channels // 2
        self.num_bins = config.duration_predictor_flow_bins
        self.tail_bound = config.duration_predictor_tail_bound
        self.conv_pre = nn.Conv1d(self.half_channels, self.filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(config)
        self.conv_proj = nn.Conv1d(self.filter_channels, self.half_channels * (self.num_bins * 3 - 1), 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)
        hidden_states = self.conv_pre(first_half)
        hidden_states = self.conv_dds(hidden_states, padding_mask, global_conditioning)
        hidden_states = self.conv_proj(hidden_states) * padding_mask
        batch_size, channels, length = first_half.shape
        hidden_states = hidden_states.reshape(batch_size, channels, -1, length).permute(0, 1, 3, 2)
        unnormalized_widths = hidden_states[..., :self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = hidden_states[..., self.num_bins:2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = hidden_states[..., 2 * self.num_bins:]
        second_half, log_abs_det = _unconstrained_rational_quadratic_spline(second_half, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, reverse=reverse, tail_bound=self.tail_bound)
        outputs = torch.cat([first_half, second_half], dim=1) * padding_mask
        if not reverse:
            log_determinant = torch.sum(log_abs_det * padding_mask, [1, 2])
            return (outputs, log_determinant)
        else:
            return (outputs, None)