import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class DenseNet169_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/densenet169-b2777c0a.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 14149480, '_metrics': {'ImageNet-1K': {'acc@1': 75.6, 'acc@5': 92.806}}, '_ops': 3.36, '_file_size': 54.708})
    DEFAULT = IMAGENET1K_V1