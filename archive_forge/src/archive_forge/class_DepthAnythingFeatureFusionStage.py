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
class DepthAnythingFeatureFusionStage(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(DepthAnythingFeatureFusionLayer(config))

    def forward(self, hidden_states, size=None):
        hidden_states = hidden_states[::-1]
        fused_hidden_states = []
        size = hidden_states[1].shape[2:]
        fused_hidden_state = self.layers[0](hidden_states[0], size=size)
        fused_hidden_states.append(fused_hidden_state)
        for idx, (hidden_state, layer) in enumerate(zip(hidden_states[1:], self.layers[1:])):
            size = hidden_states[1:][idx + 1].shape[2:] if idx != len(hidden_states[1:]) - 1 else None
            fused_hidden_state = layer(fused_hidden_state, hidden_state, size=size)
            fused_hidden_states.append(fused_hidden_state)
        return fused_hidden_states