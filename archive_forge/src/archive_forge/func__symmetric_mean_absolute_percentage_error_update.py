from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _symmetric_mean_absolute_percentage_error_update(preds: Tensor, target: Tensor, epsilon: float=1.17e-06) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Symmetric Mean Absolute Percentage Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        epsilon: Avoids ``ZeroDivisionError``.

    """
    _check_same_shape(preds, target)
    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.clamp(torch.abs(target) + torch.abs(preds), min=epsilon)
    sum_abs_per_error = 2 * torch.sum(abs_per_error)
    num_obs = target.numel()
    return (sum_abs_per_error, num_obs)