import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class EfficientNet_B3_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth', transforms=partial(ImageClassification, crop_size=300, resize_size=320, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_META_V1, 'num_params': 12233232, '_metrics': {'ImageNet-1K': {'acc@1': 82.008, 'acc@5': 96.054}}, '_ops': 1.827, '_file_size': 47.184, '_docs': 'These weights are ported from the original paper.'})
    DEFAULT = IMAGENET1K_V1