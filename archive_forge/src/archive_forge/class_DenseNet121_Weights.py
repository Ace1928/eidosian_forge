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
class DenseNet121_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/densenet121-a639ec97.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 7978856, '_metrics': {'ImageNet-1K': {'acc@1': 74.434, 'acc@5': 91.972}}, '_ops': 2.834, '_file_size': 30.845})
    DEFAULT = IMAGENET1K_V1