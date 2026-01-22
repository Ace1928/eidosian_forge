from functools import partial
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import shufflenetv2
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..shufflenetv2 import (
from .utils import _fuse_modules, _replace_relu, quantize_model
class ShuffleNet_V2_X2_0_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(url='https://download.pytorch.org/models/quantized/shufflenetv2_x2_0_fbgemm-5cac526c.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'recipe': 'https://github.com/pytorch/vision/pull/5906', 'num_params': 7393996, 'unquantized': ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1, '_metrics': {'ImageNet-1K': {'acc@1': 75.354, 'acc@5': 92.488}}, '_ops': 0.583, '_file_size': 7.467})
    DEFAULT = IMAGENET1K_FBGEMM_V1