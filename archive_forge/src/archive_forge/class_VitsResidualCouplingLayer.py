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
class VitsResidualCouplingLayer(nn.Module):

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.half_channels = config.flow_size // 2
        self.conv_pre = nn.Conv1d(self.half_channels, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=config.prior_encoder_num_wavenet_layers)
        self.conv_post = nn.Conv1d(config.hidden_size, self.half_channels, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)
        hidden_states = self.conv_pre(first_half) * padding_mask
        hidden_states = self.wavenet(hidden_states, padding_mask, global_conditioning)
        mean = self.conv_post(hidden_states) * padding_mask
        log_stddev = torch.zeros_like(mean)
        if not reverse:
            second_half = mean + second_half * torch.exp(log_stddev) * padding_mask
            outputs = torch.cat([first_half, second_half], dim=1)
            log_determinant = torch.sum(log_stddev, [1, 2])
            return (outputs, log_determinant)
        else:
            second_half = (second_half - mean) * torch.exp(-log_stddev) * padding_mask
            outputs = torch.cat([first_half, second_half], dim=1)
            return (outputs, None)