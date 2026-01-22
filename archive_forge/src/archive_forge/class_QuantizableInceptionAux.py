import warnings
from functools import partial
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..googlenet import BasicConv2d, GoogLeNet, GoogLeNet_Weights, GoogLeNetOutputs, Inception, InceptionAux
from .utils import _fuse_modules, _replace_relu, quantize_model
class QuantizableInceptionAux(InceptionAux):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, conv_block=QuantizableBasicConv2d, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x