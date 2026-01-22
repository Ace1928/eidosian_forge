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
class MaskPredictor(nn.Module):
    """Mask predictor to be used when upsampling the predicted flow.

    It takes the hidden state of the recurrent unit as input and outputs the mask.
    This is not used in the raft-small model.
    """

    def __init__(self, *, in_channels, hidden_size, multiplier=0.25):
        super().__init__()
        self.convrelu = Conv2dNormActivation(in_channels, hidden_size, norm_layer=None, kernel_size=3)
        self.conv = nn.Conv2d(hidden_size, 8 * 8 * 9, 1, padding=0)
        self.multiplier = multiplier

    def forward(self, x):
        x = self.convrelu(x)
        x = self.conv(x)
        return self.multiplier * x