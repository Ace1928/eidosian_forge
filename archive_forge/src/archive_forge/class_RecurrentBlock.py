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
class RecurrentBlock(nn.Module):
    """Recurrent block, part of the update block.

    Takes the current hidden state and the concatenation of (motion encoder output, context) as input.
    Returns an updated hidden state.
    """

    def __init__(self, *, input_size, hidden_size, kernel_size=((1, 5), (5, 1)), padding=((0, 2), (2, 0))):
        super().__init__()
        if len(kernel_size) != len(padding):
            raise ValueError(f'kernel_size should have the same length as padding, instead got len(kernel_size) = {len(kernel_size)} and len(padding) = {len(padding)}')
        if len(kernel_size) not in (1, 2):
            raise ValueError(f'kernel_size should either 1 or 2, instead got {len(kernel_size)}')
        self.convgru1 = ConvGRU(input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[0], padding=padding[0])
        if len(kernel_size) == 2:
            self.convgru2 = ConvGRU(input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[1], padding=padding[1])
        else:
            self.convgru2 = _pass_through_h
        self.hidden_size = hidden_size

    def forward(self, h, x):
        h = self.convgru1(h, x)
        h = self.convgru2(h, x)
        return h