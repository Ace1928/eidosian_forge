from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def _jaccard_index_reduce(confmat: Tensor, average: Optional[Literal['micro', 'macro', 'weighted', 'none', 'binary']], ignore_index: Optional[int]=None) -> Tensor:
    """Perform reduction of an un-normalized confusion matrix into jaccard score.

    Args:
        confmat: tensor with un-normalized confusionmatrix
        average: reduction method

            - ``'binary'``: binary reduction, expects a 2x2 matrix
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation

    """
    allowed_average = ['binary', 'micro', 'macro', 'weighted', 'none', None]
    if average not in allowed_average:
        raise ValueError(f'The `average` has to be one of {allowed_average}, got {average}.')
    confmat = confmat.float()
    if average == 'binary':
        return confmat[1, 1] / (confmat[0, 1] + confmat[1, 0] + confmat[1, 1])
    ignore_index_cond = ignore_index is not None and 0 <= ignore_index < confmat.shape[0]
    multilabel = confmat.ndim == 3
    if multilabel:
        num = confmat[:, 1, 1]
        denom = confmat[:, 1, 1] + confmat[:, 0, 1] + confmat[:, 1, 0]
    else:
        num = torch.diag(confmat)
        denom = confmat.sum(0) + confmat.sum(1) - num
    if average == 'micro':
        num = num.sum()
        denom = denom.sum() - (denom[ignore_index] if ignore_index_cond else 0.0)
    jaccard = _safe_divide(num, denom)
    if average is None or average == 'none' or average == 'micro':
        return jaccard
    if average == 'weighted':
        weights = confmat[:, 1, 1] + confmat[:, 1, 0] if confmat.ndim == 3 else confmat.sum(1)
    else:
        weights = torch.ones_like(jaccard)
        if ignore_index_cond:
            weights[ignore_index] = 0.0
        if not multilabel:
            weights[confmat.sum(1) + confmat.sum(0) == 0] = 0.0
    return (weights * jaccard / weights.sum()).sum()