from typing import Tuple
import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once
from ._box_convert import _box_cxcywh_to_xyxy, _box_xywh_to_xyxy, _box_xyxy_to_cxcywh, _box_xyxy_to_xywh
from ._utils import _upcast
def _box_diou_iou(boxes1: Tensor, boxes2: Tensor, eps: float=1e-07) -> Tuple[Tensor, Tensor]:
    iou = box_iou(boxes1, boxes2)
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    whi = _upcast(rbi - lti).clamp(min=0)
    diagonal_distance_squared = whi[:, :, 0] ** 2 + whi[:, :, 1] ** 2 + eps
    x_p = (boxes1[:, 0] + boxes1[:, 2]) / 2
    y_p = (boxes1[:, 1] + boxes1[:, 3]) / 2
    x_g = (boxes2[:, 0] + boxes2[:, 2]) / 2
    y_g = (boxes2[:, 1] + boxes2[:, 3]) / 2
    centers_distance_squared = _upcast(x_p[:, None] - x_g[None, :]) ** 2 + _upcast(y_p[:, None] - y_g[None, :]) ** 2
    return (iou - centers_distance_squared / diagonal_distance_squared, iou)