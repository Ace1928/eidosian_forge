import torch
from torch.nn import Conv1d, Conv2d, Conv3d, ReLU, Linear, BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.utils.parametrize import type_before_parametrizations
class _FusedModule(torch.nn.Sequential):
    pass