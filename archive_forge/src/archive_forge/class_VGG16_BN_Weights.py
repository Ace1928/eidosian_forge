from functools import partial
from typing import Any, cast, Dict, List, Optional, Union
import torch
import torch.nn as nn
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class VGG16_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 138365992, '_metrics': {'ImageNet-1K': {'acc@1': 73.36, 'acc@5': 91.516}}, '_ops': 15.47, '_file_size': 527.866})
    DEFAULT = IMAGENET1K_V1