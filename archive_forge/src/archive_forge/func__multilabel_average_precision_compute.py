from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multilabel_average_precision_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_labels: int, average: Optional[Literal['micro', 'macro', 'weighted', 'none']], thresholds: Optional[Tensor], ignore_index: Optional[int]=None) -> Tensor:
    if average == 'micro':
        if isinstance(state, Tensor) and thresholds is not None:
            state = state.sum(1)
        else:
            preds, target = (state[0].flatten(), state[1].flatten())
            if ignore_index is not None:
                idx = target == ignore_index
                preds = preds[~idx]
                target = target[~idx]
            state = (preds, target)
        return _binary_average_precision_compute(state, thresholds)
    precision, recall, _ = _multilabel_precision_recall_curve_compute(state, num_labels, thresholds, ignore_index)
    return _reduce_average_precision(precision, recall, average, weights=(state[1] == 1).sum(dim=0).float() if thresholds is None else state[0][:, 1, :].sum(-1))