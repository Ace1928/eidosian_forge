import warnings
from functools import partial
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class MNASNet1_0_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 4383312, '_metrics': {'ImageNet-1K': {'acc@1': 73.456, 'acc@5': 91.51}}, '_ops': 0.314, '_file_size': 16.915, '_docs': 'These weights reproduce closely the results of the paper.'})
    DEFAULT = IMAGENET1K_V1