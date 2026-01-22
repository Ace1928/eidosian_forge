import warnings
from functools import partial
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import inception as inception_module
from torchvision.models.inception import Inception_V3_Weights, InceptionOutputs
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .utils import _fuse_modules, _replace_relu, quantize_model
class QuantizableInceptionE(inception_module.InceptionE):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, conv_block=QuantizableBasicConv2d, **kwargs)
        self.myop1 = nn.quantized.FloatFunctional()
        self.myop2 = nn.quantized.FloatFunctional()
        self.myop3 = nn.quantized.FloatFunctional()

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = self.myop1.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = self.myop2.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.myop3.cat(outputs, 1)