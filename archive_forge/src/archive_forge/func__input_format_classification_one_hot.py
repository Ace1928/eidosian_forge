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
def _input_format_classification_one_hot(num_classes: int, preds: Tensor, target: Tensor, threshold: float=0.5, multilabel: bool=False) -> Tuple[Tensor, Tensor]:
    """Convert preds and target tensors into one hot spare label tensors.

    Args:
        num_classes: number of classes
        preds: either tensor with labels, tensor with probabilities/logits or multilabel tensor
        target: tensor with ground-true labels
        threshold: float used for thresholding multilabel input
        multilabel: boolean flag indicating if input is multilabel

    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same number of dimensions
            or one additional dimension for ``preds``.

    Returns:
        preds: one hot tensor of shape [num_classes, -1] with predicted labels
        target: one hot tensors of shape [num_classes, -1] with true labels

    """
    if preds.ndim not in (target.ndim, target.ndim + 1):
        raise ValueError('preds and target must have same number of dimensions, or one additional dimension for preds')
    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, dim=1)
    if preds.ndim == target.ndim and preds.dtype in (torch.long, torch.int) and (num_classes > 1) and (not multilabel):
        preds = to_onehot(preds, num_classes=num_classes)
        target = to_onehot(target, num_classes=num_classes)
    elif preds.ndim == target.ndim and preds.is_floating_point():
        preds = (preds >= threshold).long()
    if preds.ndim > 1:
        preds = preds.transpose(1, 0)
        target = target.transpose(1, 0)
    return (preds.reshape(num_classes, -1), target.reshape(num_classes, -1))