from functools import partial
from typing import Any, Callable, Optional
import torch
from torch import nn
from torchvision.ops.misc import Conv3dNormActivation
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
class S3D(nn.Module):
    """S3D main class.

    Args:
        num_class (int): number of classes for the classification task.
        dropout (float): dropout probability.
        norm_layer (Optional[Callable]): Module specifying the normalization layer to use.

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
    """

    def __init__(self, num_classes: int=400, dropout: float=0.2, norm_layer: Optional[Callable[..., torch.nn.Module]]=None) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm3d, eps=0.001, momentum=0.001)
        self.features = nn.Sequential(TemporalSeparableConv(3, 64, 7, 2, 3, norm_layer), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), Conv3dNormActivation(64, 64, kernel_size=1, stride=1, norm_layer=norm_layer), TemporalSeparableConv(64, 192, 3, 1, 1, norm_layer), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), SepInceptionBlock3D(192, 64, 96, 128, 16, 32, 32, norm_layer), SepInceptionBlock3D(256, 128, 128, 192, 32, 96, 64, norm_layer), nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), SepInceptionBlock3D(480, 192, 96, 208, 16, 48, 64, norm_layer), SepInceptionBlock3D(512, 160, 112, 224, 24, 64, 64, norm_layer), SepInceptionBlock3D(512, 128, 128, 256, 24, 64, 64, norm_layer), SepInceptionBlock3D(512, 112, 144, 288, 32, 64, 64, norm_layer), SepInceptionBlock3D(528, 256, 160, 320, 32, 128, 128, norm_layer), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), SepInceptionBlock3D(832, 256, 160, 320, 32, 128, 128, norm_layer), SepInceptionBlock3D(832, 384, 192, 384, 48, 128, 128, norm_layer))
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = torch.mean(x, dim=(2, 3, 4))
        return x