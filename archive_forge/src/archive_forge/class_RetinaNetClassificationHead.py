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
class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """
    _version = 2

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01, norm_layer: Optional[Callable[..., nn.Module]]=None):
        super().__init__()
        conv = []
        for _ in range(4):
            conv.append(misc_nn_ops.Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))
        self.conv = nn.Sequential(*conv)
        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        cls_logits = head_outputs['cls_logits']
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[foreground_idxs_per_image, targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]]] = 1.0
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS
            losses.append(sigmoid_focal_loss(cls_logits_per_image[valid_idxs_per_image], gt_classes_target[valid_idxs_per_image], reduction='sum') / max(1, num_foreground))
        return _sum(losses) / len(targets)

    def forward(self, x):
        all_cls_logits = []
        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)
            all_cls_logits.append(cls_logits)
        return torch.cat(all_cls_logits, dim=1)