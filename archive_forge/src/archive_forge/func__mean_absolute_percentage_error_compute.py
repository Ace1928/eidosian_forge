from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _mean_absolute_percentage_error_compute(sum_abs_per_error: Tensor, num_obs: Union[int, Tensor]) -> Tensor:
    """Compute Mean Absolute Percentage Error.

    Args:
        sum_abs_per_error: Sum of absolute value of percentage errors over all observations
            ``(percentage error = (target - prediction) / target)``
        num_obs: Number of predictions or observations

    Example:
        >>> target = torch.tensor([1, 10, 1e6])
        >>> preds = torch.tensor([0.9, 15, 1.2e6])
        >>> sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)
        >>> _mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)
        tensor(0.2667)

    """
    return sum_abs_per_error / num_obs