from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _mean_absolute_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    preds = preds if preds.is_floating_point else preds.float()
    target = target if target.is_floating_point else target.float()
    sum_abs_error = torch.sum(torch.abs(preds - target))
    return (sum_abs_error, target.numel())