from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.LRASPP``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights
        :members:
    