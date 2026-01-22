from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class MobileNet_V3_Small_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 2542856, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small', '_metrics': {'ImageNet-1K': {'acc@1': 67.668, 'acc@5': 87.402}}, '_ops': 0.057, '_file_size': 9.829, '_docs': '\n                These weights improve upon the results of the original paper by using a simple training recipe.\n            '})
    DEFAULT = IMAGENET1K_V1