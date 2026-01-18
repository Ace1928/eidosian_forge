from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
Compute mean squared error.

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RMSE value if set to False
        num_outputs: Number of outputs in multioutput setting

    Return:
        Tensor with MSE

    Example:
        >>> from torchmetrics.functional.regression import mean_squared_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_squared_error(x, y)
        tensor(0.2500)

    