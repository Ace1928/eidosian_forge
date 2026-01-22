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
class SqueezeNet1_0_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'min_size': (21, 21), 'num_params': 1248424, '_metrics': {'ImageNet-1K': {'acc@1': 58.092, 'acc@5': 80.42}}, '_ops': 0.819, '_file_size': 4.778})
    DEFAULT = IMAGENET1K_V1