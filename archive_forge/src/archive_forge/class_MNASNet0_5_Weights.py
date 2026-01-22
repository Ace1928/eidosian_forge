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
class MNASNet0_5_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 2218512, '_metrics': {'ImageNet-1K': {'acc@1': 67.734, 'acc@5': 87.49}}, '_ops': 0.104, '_file_size': 8.591, '_docs': 'These weights reproduce closely the results of the paper.'})
    DEFAULT = IMAGENET1K_V1