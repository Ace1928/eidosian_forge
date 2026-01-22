from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _binary_confusion_matrix_format(preds: Tensor, target: Tensor, threshold: float=0.5, ignore_index: Optional[int]=None, convert_to_labels: bool=True) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - Remove all datapoints that should be ignored
    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards

    """
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]
    if preds.is_floating_point():
        if not torch.all((preds >= 0) * (preds <= 1)):
            preds = preds.sigmoid()
        if convert_to_labels:
            preds = preds > threshold
    return (preds, target)