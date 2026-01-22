import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
class SqueezeNet(torch.nn.Module):
    """SqueezeNet implementation."""

    def __init__(self, requires_grad: bool=False, pretrained: bool=True) -> None:
        super().__init__()
        pretrained_features = _get_net('squeezenet1_1', pretrained)
        self.N_slices = 7
        slices = []
        feature_ranges = [range(2), range(2, 5), range(5, 8), range(8, 10), range(10, 11), range(11, 12), range(12, 13)]
        for feature_range in feature_ranges:
            seq = torch.nn.Sequential()
            for i in feature_range:
                seq.add_module(str(i), pretrained_features[i])
            slices.append(seq)
        self.slices = nn.ModuleList(slices)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> NamedTuple:
        """Process input."""

        class _SqueezeOutput(NamedTuple):
            relu1: Tensor
            relu2: Tensor
            relu3: Tensor
            relu4: Tensor
            relu5: Tensor
            relu6: Tensor
            relu7: Tensor
        relus = []
        for slice_ in self.slices:
            x = slice_(x)
            relus.append(x)
        return _SqueezeOutput(*relus)