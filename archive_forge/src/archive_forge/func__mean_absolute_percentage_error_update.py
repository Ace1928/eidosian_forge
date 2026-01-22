from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _mean_absolute_percentage_error_update(preds: Tensor, target: Tensor, epsilon: float=1.17e-06) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Percentage Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        epsilon: Specifies the lower bound for target values. Any target value below epsilon
            is set to epsilon (avoids ``ZeroDivisionError``).

    """
    _check_same_shape(preds, target)
    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.clamp(torch.abs(target), min=epsilon)
    sum_abs_per_error = torch.sum(abs_per_error)
    num_obs = target.numel()
    return (sum_abs_per_error, num_obs)