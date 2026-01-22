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
class PatchTSMixerLinearHead(nn.Module):
    """Linear head for Classification and Regression.

    Args:
        config (`PatchTSMixerConfig`, *required*):

    """

    def __init__(self, config: PatchTSMixerConfig, distribution_output=None):
        super().__init__()
        self.head_aggregation = config.head_aggregation
        self.output_range = config.output_range
        if config.head_aggregation is None:
            mul_factor = config.num_patches
        else:
            mul_factor = 1
        self.distribution_output = distribution_output
        if distribution_output is None:
            self.projection = nn.Linear(config.d_model * config.num_input_channels * mul_factor, config.num_targets)
        else:
            self.projection = distribution_output.get_parameter_projection(config.d_model * config.num_input_channels * mul_factor)
        if config.head_aggregation is None:
            self.flatten = nn.Flatten(start_dim=-3)
        else:
            self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(config.head_dropout)

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x num_targets)`.
        """
        hidden_features = hidden_features.transpose(-1, -2)
        if self.head_aggregation == 'use_last':
            hidden_features = hidden_features[..., -1]
        elif self.head_aggregation == 'max_pool':
            hidden_features = hidden_features.max(dim=-1).values
        elif self.head_aggregation == 'avg_pool':
            hidden_features = hidden_features.mean(dim=-1)
        if self.flatten:
            hidden_features = self.flatten(hidden_features)
        hidden_features = self.dropout(hidden_features)
        hidden_features = self.projection(hidden_features)
        if self.distribution_output is None and self.output_range is not None:
            hidden_features = torch.sigmoid(hidden_features) * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        return hidden_features