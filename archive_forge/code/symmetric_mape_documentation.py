from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
Compute symmetric mean absolute percentage error (SMAPE_).

    .. math:: \text{SMAPE} = \frac{2}{n}\sum_1^n\frac{|   y_i - \hat{y_i} |}{max(| y_i | + | \hat{y_i} |, \epsilon)}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with SMAPE.

    Example:
        >>> from torchmetrics.functional.regression import symmetric_mean_absolute_percentage_error
        >>> target = torch.tensor([1, 10, 1e6])
        >>> preds = torch.tensor([0.9, 15, 1.2e6])
        >>> symmetric_mean_absolute_percentage_error(preds, target)
        tensor(0.2290)

    