import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
class PatchTSTScaler(nn.Module):

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        if config.scaling == 'mean' or config.scaling is True:
            self.scaler = PatchTSTMeanScaler(config)
        elif config.scaling == 'std':
            self.scaler = PatchTSTStdScaler(config)
        else:
            self.scaler = PatchTSTNOPScaler(config)

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Input for scaler calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, um_input_channels)`)
        """
        data, loc, scale = self.scaler(data, observed_indicator)
        return (data, loc, scale)