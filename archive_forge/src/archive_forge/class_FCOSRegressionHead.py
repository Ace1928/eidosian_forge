import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ...ops import boxes as box_ops, generalized_box_iou_loss, misc as misc_nn_ops, sigmoid_focal_loss
from ...ops.feature_pyramid_network import LastLevelP6P7
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from . import _utils as det_utils
from .anchor_utils import AnchorGenerator
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
class FCOSRegressionHead(nn.Module):
    """
    A regression head for use in FCOS, which combines regression branch and center-ness branch.
    This can obtain better performance.

    Reference: `FCOS: A simple and strong anchor-free object detector <https://arxiv.org/abs/2006.09214>`_.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 4.
        norm_layer: Module specifying the normalization layer to use.
    """

    def __init__(self, in_channels: int, num_anchors: int, num_convs: int=4, norm_layer: Optional[Callable[..., nn.Module]]=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)
        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)
        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.bbox_ctrness = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1)
        for layer in [self.bbox_reg, self.bbox_ctrness]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.zeros_(layer.bias)
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        all_bbox_regression = []
        all_bbox_ctrness = []
        for features in x:
            bbox_feature = self.conv(features)
            bbox_regression = nn.functional.relu(self.bbox_reg(bbox_feature))
            bbox_ctrness = self.bbox_ctrness(bbox_feature)
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)
            all_bbox_regression.append(bbox_regression)
            bbox_ctrness = bbox_ctrness.view(N, -1, 1, H, W)
            bbox_ctrness = bbox_ctrness.permute(0, 3, 4, 1, 2)
            bbox_ctrness = bbox_ctrness.reshape(N, -1, 1)
            all_bbox_ctrness.append(bbox_ctrness)
        return (torch.cat(all_bbox_regression, dim=1), torch.cat(all_bbox_ctrness, dim=1))