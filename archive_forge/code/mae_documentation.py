from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
Compute mean absolute error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with MAE

    Example:
        >>> from torchmetrics.functional.regression import mean_absolute_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_absolute_error(x, y)
        tensor(0.2500)

    