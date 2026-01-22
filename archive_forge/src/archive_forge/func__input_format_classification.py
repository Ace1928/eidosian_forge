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
def _input_format_classification(preds: Tensor, target: Tensor, threshold: float=0.5, top_k: Optional[int]=None, num_classes: Optional[int]=None, multiclass: Optional[bool]=None, ignore_index: Optional[int]=None) -> Tuple[Tensor, Tensor, DataType]:
    """Convert preds and target tensors into common format.

    Preds and targets are supposed to fall into one of these categories (and are
    validated to make sure this is the case):

        * Both preds and target are of shape ``(N,)``, and both are integers (multi-class)
        * Both preds and target are of shape ``(N,)``, and target is binary, while preds
          are a float (binary)
        * preds are of shape ``(N, C)`` and are floats, and target is of shape ``(N,)`` and
          is integer (multi-class)
        * preds and target are of shape ``(N, ...)``, target is binary and preds is a float
          (multi-label)
        * preds are of shape ``(N, C, ...)`` and are floats, target is of shape ``(N, ...)``
          and is integer (multi-dimensional multi-class)
        * preds and target are of shape ``(N, ...)`` both are integers (multi-dimensional
          multi-class)

    To avoid ambiguities, all dimensions of size 1, except the first one, are squeezed out.

    The returned output tensors will be binary tensors of the same shape, either ``(N, C)``
    of ``(N, C, X)``, the details for each case are described below. The function also returns
    a ``case`` string, which describes which of the above cases the inputs belonged to - regardless
    of whether this was "overridden" by other settings (like ``multiclass``).

    In binary case, targets are normally returned as ``(N,1)`` tensor, while preds are transformed
    into a binary tensor (elements become 1 if the probability is greater than or equal to
    ``threshold`` or 0 otherwise). If ``multiclass=True``, then both targets are preds
    become ``(N, 2)`` tensors by a one-hot transformation; with the thresholding being applied to
    preds first.

    In multi-class case, normally both preds and targets become ``(N, C)`` binary tensors; targets
    by a one-hot transformation and preds by selecting ``top_k`` largest entries (if their original
    shape was ``(N,C)``). However, if ``multiclass=False``, then targets and preds will be
    returned as ``(N,1)`` tensor.

    In multi-label case, normally targets and preds are returned as ``(N, C)`` binary tensors, with
    preds being binarized as in the binary case. Here the ``C`` dimension is obtained by flattening
    all dimensions after the first one. However, if ``multiclass=True``, then both are returned as
    ``(N, 2, C)``, by an equivalent transformation as in the binary case.

    In multi-dimensional multi-class case, normally both target and preds are returned as
    ``(N, C, X)`` tensors, with ``X`` resulting from flattening of all dimensions except ``N`` and
    ``C``. The transformations performed here are equivalent to the multi-class case. However, if
    ``multiclass=False`` (and there are up to two classes), then the data is returned as
    ``(N, X)`` binary tensors (multi-label).

    Note:
        Where a one-hot transformation needs to be performed and the number of classes
        is not implicitly given by a ``C`` dimension, the new ``C`` dimension will either be
        equal to ``num_classes``, if it is given, or the maximum label value in preds and
        target.

    Args:
        preds: Tensor with predictions (labels or probabilities)
        target: Tensor with ground truth labels, always integers (labels)
        threshold:
            Threshold value for transforming probability/logit predictions to binary
            (0 or 1) predictions, in the case of binary or multi-label inputs.
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be inferred
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of the highest probability entries for each sample to convert to 1s - relevant
            only for (multi-dimensional) multi-class inputs with probability predictions. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left unset (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/overview:using the multiclass parameter>`
            for a more detailed explanation and examples.
        ignore_index: ignore predictions where targets are equal to this number

    Returns:
        preds: binary tensor of shape ``(N, C)`` or ``(N, C, X)``
        target: binary tensor of shape ``(N, C)`` or ``(N, C, X)``
        case: The case the inputs fall in, one of ``'binary'``, ``'multi-class'``, ``'multi-label'`` or
            ``'multi-dim multi-class'``

    """
    preds, target = _input_squeeze(preds, target)
    if preds.dtype == torch.float16:
        preds = preds.float()
    case = _check_classification_inputs(preds, target, threshold=threshold, num_classes=num_classes, multiclass=multiclass, top_k=top_k, ignore_index=ignore_index)
    if case in (DataType.BINARY, DataType.MULTILABEL) and (not top_k):
        preds = (preds >= threshold).int()
        num_classes = num_classes if not multiclass else 2
    if case == DataType.MULTILABEL and top_k:
        preds = select_topk(preds, top_k)
    if case in (DataType.MULTICLASS, DataType.MULTIDIM_MULTICLASS) or multiclass:
        if preds.is_floating_point():
            num_classes = preds.shape[1]
            preds = select_topk(preds, top_k or 1)
        else:
            num_classes = num_classes or int(max(preds.max().item(), target.max().item()) + 1)
            preds = to_onehot(preds, max(2, num_classes))
        target = to_onehot(target, max(2, num_classes))
        if multiclass is False:
            preds, target = (preds[:, 1, ...], target[:, 1, ...])
    if not _check_for_empty_tensors(preds, target):
        if case in (DataType.MULTICLASS, DataType.MULTIDIM_MULTICLASS) and multiclass is not False or multiclass:
            target = target.reshape(target.shape[0], target.shape[1], -1)
            preds = preds.reshape(preds.shape[0], preds.shape[1], -1)
        else:
            target = target.reshape(target.shape[0], -1)
            preds = preds.reshape(preds.shape[0], -1)
    if preds.ndim > 2:
        preds, target = (preds.squeeze(-1), target.squeeze(-1))
    return (preds.int(), target.int(), case)