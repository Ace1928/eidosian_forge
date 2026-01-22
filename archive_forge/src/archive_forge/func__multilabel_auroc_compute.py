from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.compute import _auc_compute_without_check, _safe_divide
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _multilabel_auroc_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_labels: int, average: Optional[Literal['micro', 'macro', 'weighted', 'none']], thresholds: Optional[Tensor], ignore_index: Optional[int]=None) -> Tensor:
    if average == 'micro':
        if isinstance(state, Tensor) and thresholds is not None:
            return _binary_auroc_compute(state.sum(1), thresholds, max_fpr=None)
        preds = state[0].flatten()
        target = state[1].flatten()
        if ignore_index is not None:
            idx = target == ignore_index
            preds = preds[~idx]
            target = target[~idx]
        return _binary_auroc_compute((preds, target), thresholds, max_fpr=None)
    fpr, tpr, _ = _multilabel_roc_compute(state, num_labels, thresholds, ignore_index)
    return _reduce_auroc(fpr, tpr, average, weights=(state[1] == 1).sum(dim=0).float() if thresholds is None else state[0][:, 1, :].sum(-1))