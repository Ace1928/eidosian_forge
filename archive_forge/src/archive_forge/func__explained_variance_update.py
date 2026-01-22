from typing import Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def _explained_variance_update(preds: Tensor, target: Tensor) -> Tuple[int, Tensor, Tensor, Tensor, Tensor]:
    """Update and returns variables required to compute Explained Variance. Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    num_obs = preds.size(0)
    sum_error = torch.sum(target - preds, dim=0)
    diff = target - preds
    sum_squared_error = torch.sum(diff * diff, dim=0)
    sum_target = torch.sum(target, dim=0)
    sum_squared_target = torch.sum(target * target, dim=0)
    return (num_obs, sum_error, sum_squared_error, sum_target, sum_squared_target)