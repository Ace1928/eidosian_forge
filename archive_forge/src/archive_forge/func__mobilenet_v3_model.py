from functools import partial
from typing import Any, List, Optional, Union
import torch
from torch import nn, Tensor
from torch.ao.quantization import DeQuantStub, QuantStub
from ...ops.misc import Conv2dNormActivation, SqueezeExcitation
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..mobilenetv3 import (
from .utils import _fuse_modules, _replace_relu
def _mobilenet_v3_model(inverted_residual_setting: List[InvertedResidualConfig], last_channel: int, weights: Optional[WeightsEnum], progress: bool, quantize: bool, **kwargs: Any) -> QuantizableMobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
        if 'backend' in weights.meta:
            _ovewrite_named_param(kwargs, 'backend', weights.meta['backend'])
    backend = kwargs.pop('backend', 'qnnpack')
    model = QuantizableMobileNetV3(inverted_residual_setting, last_channel, block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)
    if quantize:
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
        torch.ao.quantization.prepare_qat(model, inplace=True)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    if quantize:
        torch.ao.quantization.convert(model, inplace=True)
        model.eval()
    return model