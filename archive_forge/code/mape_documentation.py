from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
Compute mean absolute percentage error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with MAPE

    Note:
        The epsilon value is taken from `scikit-learn's implementation of MAPE`_.

    Example:
        >>> from torchmetrics.functional.regression import mean_absolute_percentage_error
        >>> target = torch.tensor([1, 10, 1e6])
        >>> preds = torch.tensor([0.9, 15, 1.2e6])
        >>> mean_absolute_percentage_error(preds, target)
        tensor(0.2667)

    