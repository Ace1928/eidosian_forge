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
class DepthAnythingDepthEstimationHead(nn.Module):
    """
    Output head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the DPT paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()
        self.head_in_index = config.head_in_index
        self.patch_size = config.patch_size
        features = config.fusion_hidden_size
        self.conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features // 2, config.head_hidden_size, kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.ReLU()
        self.conv3 = nn.Conv2d(config.head_hidden_size, 1, kernel_size=1, stride=1, padding=0)
        self.activation2 = nn.ReLU()

    def forward(self, hidden_states: List[torch.Tensor], patch_height, patch_width) -> torch.Tensor:
        hidden_states = hidden_states[self.head_in_index]
        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(predicted_depth, (int(patch_height * self.patch_size), int(patch_width * self.patch_size)), mode='bilinear', align_corners=True)
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth)
        predicted_depth = predicted_depth.squeeze(dim=1)
        return predicted_depth