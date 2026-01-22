from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ..ops.misc import Conv2dNormActivation, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class ConvNeXt_Tiny_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/convnext_tiny-983f1562.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=236), meta={**_COMMON_META, 'num_params': 28589128, '_metrics': {'ImageNet-1K': {'acc@1': 82.52, 'acc@5': 96.146}}, '_ops': 4.456, '_file_size': 109.119})
    DEFAULT = IMAGENET1K_V1