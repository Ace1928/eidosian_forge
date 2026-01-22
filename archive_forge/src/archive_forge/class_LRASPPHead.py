from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
class LRASPPHead(nn.Module):

    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(nn.Conv2d(high_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True))
        self.scale = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(high_channels, inter_channels, 1, bias=False), nn.Sigmoid())
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input['low']
        high = input['high']
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)
        return self.low_classifier(low) + self.high_classifier(x)