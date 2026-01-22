from functools import partial
from typing import Any, Optional
import torch
import torch.nn as nn
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class AlexNet_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/alexnet-owt-7be5be79.pth', transforms=partial(ImageClassification, crop_size=224), meta={'num_params': 61100840, 'min_size': (63, 63), 'categories': _IMAGENET_CATEGORIES, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg', '_metrics': {'ImageNet-1K': {'acc@1': 56.522, 'acc@5': 79.066}}, '_ops': 0.714, '_file_size': 233.087, '_docs': '\n                These weights reproduce closely the results of the paper using a simplified training recipe.\n            '})
    DEFAULT = IMAGENET1K_V1