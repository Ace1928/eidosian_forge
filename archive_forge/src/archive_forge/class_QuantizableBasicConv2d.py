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
class QuantizableBasicConv2d(BasicConv2d):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def fuse_model(self, is_qat: Optional[bool]=None) -> None:
        _fuse_modules(self, ['conv', 'bn', 'relu'], is_qat, inplace=True)