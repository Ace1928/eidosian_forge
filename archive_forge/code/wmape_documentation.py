from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
Compute weighted mean absolute percentage error (`WMAPE`_).

    The output of WMAPE metric is a non-negative floating point, where the optimal value is 0. It is computes as:

    .. math::
        \text{WMAPE} = \frac{\sum_{t=1}^n | y_t - \hat{y}_t | }{\sum_{t=1}^n |y_t| }

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with WMAPE.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randn(20,)
        >>> target = torch.randn(20,)
        >>> weighted_mean_absolute_percentage_error(preds, target)
        tensor(1.3967)

    