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
class ResNeXt101_64X4D_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(url='https://download.pytorch.org/models/quantized/resnext101_64x4d_fbgemm-605a1cb3.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'num_params': 83455272, 'recipe': 'https://github.com/pytorch/vision/pull/5935', 'unquantized': ResNeXt101_64X4D_Weights.IMAGENET1K_V1, '_metrics': {'ImageNet-1K': {'acc@1': 82.898, 'acc@5': 96.326}}, '_ops': 15.46, '_file_size': 81.556})
    DEFAULT = IMAGENET1K_FBGEMM_V1