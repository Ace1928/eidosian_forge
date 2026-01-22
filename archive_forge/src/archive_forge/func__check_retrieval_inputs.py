import multiprocessing
import os
import sys
from functools import partial
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, no_type_check
from unittest.mock import Mock
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import select_topk, to_onehot
from torchmetrics.utilities.enums import DataType
def _check_retrieval_inputs(indexes: Tensor, preds: Tensor, target: Tensor, allow_non_binary_target: bool=False, ignore_index: Optional[int]=None) -> Tuple[Tensor, Tensor, Tensor]:
    """Check ``indexes``, ``preds`` and ``target`` tensors are of the same shape and of the correct data type.

    Args:
        indexes: tensor with queries indexes
        preds: tensor with scores/logits
        target: tensor with ground true labels
        allow_non_binary_target: whether to allow target to contain non-binary values
        ignore_index: ignore predictions where targets are equal to this number

    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same shape, if they are empty or not of the correct ``dtypes``.

    Returns:
        indexes: as ``torch.long``
        preds: as ``torch.float32``
        target: as ``torch.long``

    """
    if indexes.shape != preds.shape or preds.shape != target.shape:
        raise ValueError('`indexes`, `preds` and `target` must be of the same shape')
    if indexes.dtype is not torch.long:
        raise ValueError('`indexes` must be a tensor of long integers')
    if ignore_index is not None:
        valid_positions = target != ignore_index
        indexes, preds, target = (indexes[valid_positions], preds[valid_positions], target[valid_positions])
    if not indexes.numel() or not indexes.size():
        raise ValueError('`indexes`, `preds` and `target` must be non-empty and non-scalar tensors')
    preds, target = _check_retrieval_target_and_prediction_types(preds, target, allow_non_binary_target=allow_non_binary_target)
    return (indexes.long().flatten(), preds, target)