from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
Compute mean squared log error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with RMSLE

    Example:
        >>> from torchmetrics.functional.regression import mean_squared_log_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_squared_log_error(x, y)
        tensor(0.0207)

    .. note::
        Half precision is only support on GPU for this metric

    