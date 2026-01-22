import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_patchtsmixer import PatchTSMixerConfig
class InjectScalerStatistics4D(nn.Module):

    def __init__(self, d_model: int, num_patches: int, expansion: int=2):
        super().__init__()
        self.inverse_trans_expansion = nn.Linear(d_model + 2, expansion * d_model)
        self.inverse_trans_compression = nn.Linear(expansion * d_model, d_model)
        self.map_scale_expansion = nn.Linear(2, 2 * expansion)
        self.map_scale_compression = nn.Linear(2 * expansion, 2)
        self.num_patches = num_patches

    def forward(self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`)
            loc (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
            scale (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
        Returns:
            `torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`
        """
        mean = loc.transpose(-1, -2)
        mean = mean.unsqueeze(-2)
        mean = mean.repeat(1, 1, self.num_patches, 1)
        stdev = scale.transpose(-1, -2)
        stdev = stdev.unsqueeze(-2)
        stdev = stdev.repeat(1, 1, self.num_patches, 1)
        concat_stats = torch.cat([mean, stdev], dim=-1)
        concat_stats = self.map_scale_expansion(concat_stats)
        concat_stats = self.map_scale_compression(concat_stats)
        inputs = torch.cat([inputs, concat_stats], dim=-1)
        inputs = self.inverse_trans_expansion(inputs)
        inputs = self.inverse_trans_compression(inputs)
        return inputs