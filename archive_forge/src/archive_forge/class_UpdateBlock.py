from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
class UpdateBlock(nn.Module):
    """The update block which contains the motion encoder, the recurrent block, and the flow head.

    It must expose a ``hidden_state_size`` attribute which is the hidden state size of its recurrent block.
    """

    def __init__(self, *, motion_encoder, recurrent_block, flow_head):
        super().__init__()
        self.motion_encoder = motion_encoder
        self.recurrent_block = recurrent_block
        self.flow_head = flow_head
        self.hidden_state_size = recurrent_block.hidden_size

    def forward(self, hidden_state, context, corr_features, flow):
        motion_features = self.motion_encoder(flow, corr_features)
        x = torch.cat([context, motion_features], dim=1)
        hidden_state = self.recurrent_block(hidden_state, x)
        delta_flow = self.flow_head(hidden_state)
        return (hidden_state, delta_flow)