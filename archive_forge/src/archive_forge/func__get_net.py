import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
def _get_net(net: str, pretrained: bool) -> nn.modules.container.Sequential:
    """Get torchvision network.

    Args:
        net: Name of network
        pretrained: If pretrained weights should be used

    """
    from torchvision import models as tv
    if _TORCHVISION_GREATER_EQUAL_0_13:
        if pretrained:
            pretrained_features = getattr(tv, net)(weights=getattr(tv, _weight_map[net]).IMAGENET1K_V1).features
        else:
            pretrained_features = getattr(tv, net)(weights=None).features
    else:
        pretrained_features = getattr(tv, net)(pretrained=pretrained).features
    return pretrained_features