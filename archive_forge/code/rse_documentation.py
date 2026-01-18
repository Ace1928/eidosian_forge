from typing import Union
import torch
from torch import Tensor
from torchmetrics.functional.regression.r2 import _r2_score_update
Computes the relative squared error (RSE).

    .. math:: \text{RSE} = \frac{\sum_i^N(y_i - \hat{y_i})^2}{\sum_i^N(y_i - \overline{y})^2}

    Where :math:`y` is a tensor of target values with mean :math:`\overline{y}`, and
    :math:`\hat{y}` is a tensor of predictions.

    If `preds` and `targets` are 2D tensors, the RSE is averaged over the second dim.

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RRSE value if set to False
    Return:
        Tensor with RSE

    Example:
        >>> from torchmetrics.functional.regression import relative_squared_error
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> relative_squared_error(preds, target)
        tensor(0.0514)

    