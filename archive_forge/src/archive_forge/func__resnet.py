from functools import partial
from typing import Any, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import (
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .utils import _fuse_modules, _replace_relu, quantize_model
def _resnet(block: Type[Union[QuantizableBasicBlock, QuantizableBottleneck]], layers: List[int], weights: Optional[WeightsEnum], progress: bool, quantize: bool, **kwargs: Any) -> QuantizableResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
        if 'backend' in weights.meta:
            _ovewrite_named_param(kwargs, 'backend', weights.meta['backend'])
    backend = kwargs.pop('backend', 'fbgemm')
    model = QuantizableResNet(block, layers, **kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model