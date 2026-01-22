from functools import partial
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.init as init
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class SqueezeNet1_1_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'min_size': (17, 17), 'num_params': 1235496, '_metrics': {'ImageNet-1K': {'acc@1': 58.178, 'acc@5': 80.624}}, '_ops': 0.349, '_file_size': 4.729})
    DEFAULT = IMAGENET1K_V1