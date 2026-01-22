from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multilabel_confusion_matrix_format(preds: Tensor, target: Tensor, num_labels: int, threshold: float=0.5, ignore_index: Optional[int]=None, should_threshold: bool=True) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    - Mask all elements that should be ignored with negative numbers for later filtration

    """
    if preds.is_floating_point():
        if not torch.all((preds >= 0) * (preds <= 1)):
            preds = preds.sigmoid()
        if should_threshold:
            preds = preds > threshold
    preds = torch.movedim(preds, 1, -1).reshape(-1, num_labels)
    target = torch.movedim(target, 1, -1).reshape(-1, num_labels)
    if ignore_index is not None:
        preds = preds.clone()
        target = target.clone()
        idx = target == ignore_index
        preds[idx] = -4 * num_labels
        target[idx] = -4 * num_labels
    return (preds, target)