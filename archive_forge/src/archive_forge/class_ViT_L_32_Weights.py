import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn as nn
from ..ops.misc import Conv2dNormActivation, MLP
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class ViT_L_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/vit_l_32-c7638314.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 306535400, 'min_size': (224, 224), 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#vit_l_32', '_metrics': {'ImageNet-1K': {'acc@1': 76.972, 'acc@5': 93.07}}, '_ops': 15.378, '_file_size': 1169.449, '_docs': "\n                These weights were trained from scratch by using a modified version of `DeIT\n                <https://arxiv.org/abs/2012.12877>`_'s training recipe.\n            "})
    DEFAULT = IMAGENET1K_V1