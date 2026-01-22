from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import _cumsum
def _multilabel_ranking_tensor_validation(preds: Tensor, target: Tensor, num_labels: int, ignore_index: Optional[int]=None) -> None:
    _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(f'Expected preds tensor to be floating point, but received input with dtype {preds.dtype}')