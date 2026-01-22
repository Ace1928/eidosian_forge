from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class Wide_ResNet50_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 68883240, 'recipe': 'https://github.com/pytorch/vision/pull/912#issue-445437439', '_metrics': {'ImageNet-1K': {'acc@1': 78.468, 'acc@5': 94.086}}, '_ops': 11.398, '_file_size': 131.82, '_docs': 'These weights reproduce closely the results of the paper using a simple training recipe.'})
    IMAGENET1K_V2 = Weights(url='https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'num_params': 68883240, 'recipe': 'https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres', '_metrics': {'ImageNet-1K': {'acc@1': 81.602, 'acc@5': 95.758}}, '_ops': 11.398, '_file_size': 263.124, '_docs': "\n                These weights improve upon the results of the original paper by using TorchVision's `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    DEFAULT = IMAGENET1K_V2