from typing import Optional
import torch
from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_8
def _iou_update(preds: torch.Tensor, target: torch.Tensor, iou_threshold: Optional[float], replacement_val: float=0) -> torch.Tensor:
    from torchvision.ops import box_iou
    iou = box_iou(preds, target)
    if iou_threshold is not None:
        iou[iou < iou_threshold] = replacement_val
    return iou