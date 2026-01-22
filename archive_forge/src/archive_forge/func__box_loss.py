import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, FrozenBatchNorm2d, generalized_box_iou_loss
def _box_loss(type: str, box_coder: BoxCoder, anchors_per_image: Tensor, matched_gt_boxes_per_image: Tensor, bbox_regression_per_image: Tensor, cnf: Optional[Dict[str, float]]=None) -> Tensor:
    torch._assert(type in ['l1', 'smooth_l1', 'ciou', 'diou', 'giou'], f'Unsupported loss: {type}')
    if type == 'l1':
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        return F.l1_loss(bbox_regression_per_image, target_regression, reduction='sum')
    elif type == 'smooth_l1':
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        beta = cnf['beta'] if cnf is not None and 'beta' in cnf else 1.0
        return F.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction='sum', beta=beta)
    else:
        bbox_per_image = box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
        eps = cnf['eps'] if cnf is not None and 'eps' in cnf else 1e-07
        if type == 'ciou':
            return complete_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction='sum', eps=eps)
        if type == 'diou':
            return distance_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction='sum', eps=eps)
        return generalized_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction='sum', eps=eps)