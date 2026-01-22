from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...file_utils import (
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..auto import AutoBackbone
from .configuration_depth_anything import DepthAnythingConfig
class DepthAnythingFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()
        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)
        self.residual_layer1 = DepthAnythingPreActResidualLayer(config)
        self.residual_layer2 = DepthAnythingPreActResidualLayer(config)

    def forward(self, hidden_state, residual=None, size=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode='bilinear', align_corners=False)
            hidden_state = hidden_state + self.residual_layer1(residual)
        hidden_state = self.residual_layer2(hidden_state)
        modifier = {'scale_factor': 2} if size is None else {'size': size}
        hidden_state = nn.functional.interpolate(hidden_state, **modifier, mode='bilinear', align_corners=True)
        hidden_state = self.projection(hidden_state)
        return hidden_state