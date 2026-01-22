import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class GoogLeNet_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/googlenet-1378be20.pth', transforms=partial(ImageClassification, crop_size=224), meta={'num_params': 6624904, 'min_size': (15, 15), 'categories': _IMAGENET_CATEGORIES, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#googlenet', '_metrics': {'ImageNet-1K': {'acc@1': 69.778, 'acc@5': 89.53}}, '_ops': 1.498, '_file_size': 49.731, '_docs': 'These weights are ported from the original paper.'})
    DEFAULT = IMAGENET1K_V1