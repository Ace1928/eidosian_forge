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
class VitsElementwiseAffine(nn.Module):

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.channels = config.depth_separable_channels
        self.translate = nn.Parameter(torch.zeros(self.channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(self.channels, 1))

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            outputs = self.translate + torch.exp(self.log_scale) * inputs
            outputs = outputs * padding_mask
            log_determinant = torch.sum(self.log_scale * padding_mask, [1, 2])
            return (outputs, log_determinant)
        else:
            outputs = (inputs - self.translate) * torch.exp(-self.log_scale) * padding_mask
            return (outputs, None)