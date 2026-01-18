from functools import partial
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.init as init
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.

    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        weights (:class:`~torchvision.models.SqueezeNet1_1_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.SqueezeNet1_1_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.SqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.SqueezeNet1_1_Weights
        :members:
    