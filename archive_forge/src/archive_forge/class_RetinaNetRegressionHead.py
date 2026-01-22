import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ...ops import boxes as box_ops, misc as misc_nn_ops, sigmoid_focal_loss
from ...ops.feature_pyramid_network import LastLevelP6P7
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from . import _utils as det_utils
from ._utils import _box_loss, overwrite_eps
from .anchor_utils import AnchorGenerator
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """
    _version = 2
    __annotations__ = {'box_coder': det_utils.BoxCoder}

    def __init__(self, in_channels, num_anchors, norm_layer: Optional[Callable[..., nn.Module]]=None):
        super().__init__()
        conv = []
        for _ in range(4):
            conv.append(misc_nn_ops.Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))
        self.conv = nn.Sequential(*conv)
        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)
        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self._loss_type = 'l1'

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        losses = []
        bbox_regression = head_outputs['bbox_regression']
        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(targets, bbox_regression, anchors, matched_idxs):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            losses.append(_box_loss(self._loss_type, self.box_coder, anchors_per_image, matched_gt_boxes_per_image, bbox_regression_per_image) / max(1, num_foreground))
        return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        all_bbox_regression = []
        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)
            all_bbox_regression.append(bbox_regression)
        return torch.cat(all_bbox_regression, dim=1)