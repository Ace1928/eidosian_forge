from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _reduce_average_precision(precision: Union[Tensor, List[Tensor]], recall: Union[Tensor, List[Tensor]], average: Optional[Literal['macro', 'weighted', 'none']]='macro', weights: Optional[Tensor]=None) -> Tensor:
    """Reduce multiple average precision score into one number."""
    if isinstance(precision, Tensor) and isinstance(recall, Tensor):
        res = -torch.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, :-1], 1)
    else:
        res = torch.stack([-torch.sum((r[1:] - r[:-1]) * p[:-1]) for p, r in zip(precision, recall)])
    if average is None or average == 'none':
        return res
    if torch.isnan(res).any():
        rank_zero_warn(f'Average precision score for one or more classes was `nan`. Ignoring these classes in {average}-average', UserWarning)
    idx = ~torch.isnan(res)
    if average == 'macro':
        return res[idx].mean()
    if average == 'weighted' and weights is not None:
        weights = _safe_divide(weights[idx], weights[idx].sum())
        return (res[idx] * weights).sum()
    raise ValueError('Received an incompatible combinations of inputs to make reduction.')