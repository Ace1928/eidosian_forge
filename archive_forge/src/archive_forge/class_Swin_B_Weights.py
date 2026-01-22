import math
from functools import partial
from typing import Any, Callable, List, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import MLP, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class Swin_B_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/swin_b-68c6b09e.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=238, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_META, 'num_params': 87768224, 'min_size': (224, 224), 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#swintransformer', '_metrics': {'ImageNet-1K': {'acc@1': 83.582, 'acc@5': 96.64}}, '_ops': 15.431, '_file_size': 335.364, '_docs': 'These weights reproduce closely the results of the paper using a similar training recipe.'})
    DEFAULT = IMAGENET1K_V1