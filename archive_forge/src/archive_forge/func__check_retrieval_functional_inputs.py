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
def _check_retrieval_functional_inputs(preds: Tensor, target: Tensor, allow_non_binary_target: bool=False) -> Tuple[Tensor, Tensor]:
    """Check ``preds`` and ``target`` tensors are of the same shape and of the correct data type.

    Args:
        preds: either tensor with scores/logits
        target: tensor with ground true labels
        allow_non_binary_target: whether to allow target to contain non-binary values

    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same shape, if they are empty
            or not of the correct ``dtypes``.

    Returns:
        preds: as torch.float32
        target: as torch.long if not floating point else torch.float32

    """
    if preds.shape != target.shape:
        raise ValueError('`preds` and `target` must be of the same shape')
    if not preds.numel() or not preds.size():
        raise ValueError('`preds` and `target` must be non-empty and non-scalar tensors')
    return _check_retrieval_target_and_prediction_types(preds, target, allow_non_binary_target=allow_non_binary_target)