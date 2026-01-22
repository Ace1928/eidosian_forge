from typing import Optional
import torch
from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_8
def _giou_compute(iou: torch.Tensor, aggregate: bool=True) -> torch.Tensor:
    if not aggregate:
        return iou
    return iou.diag().mean() if iou.numel() > 0 else torch.tensor(0.0, device=iou.device)