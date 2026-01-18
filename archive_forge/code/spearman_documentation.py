from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
Compute `spearmans rank correlation coefficient`_.

    .. math:
        r_s = = \frac{cov(rg_x, rg_y)}{\sigma_{rg_x} * \sigma_{rg_y}}

    where :math:`rg_x` and :math:`rg_y` are the rank associated to the variables x and y. Spearmans correlations
    coefficient corresponds to the standard pearsons correlation coefficient calculated on the rank variables.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example (single output regression):
        >>> from torchmetrics.functional.regression import spearman_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> spearman_corrcoef(preds, target)
        tensor(1.0000)

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import spearman_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> spearman_corrcoef(preds, target)
        tensor([1.0000, 1.0000])

    