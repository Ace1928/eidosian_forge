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
def _check_classification_inputs(preds: Tensor, target: Tensor, threshold: float, num_classes: Optional[int], multiclass: Optional[bool], top_k: Optional[int], ignore_index: Optional[int]=None) -> DataType:
    """Perform error checking on inputs for classification.

    This ensures that preds and target take one of the shape/type combinations that are
    specified in ``_input_format_classification`` docstring. It also checks the cases of
    over-rides with ``multiclass`` by checking (for multi-class and multi-dim multi-class
    cases) that there are only up to 2 distinct labels.

    In case where preds are floats (probabilities), it is checked whether they are in ``[0,1]`` interval.

    When ``num_classes`` is given, it is checked that it is consistent with input cases (binary,
    multi-label, ...), and that, if available, the implied number of classes in the ``C``
    dimension is consistent with it (as well as that max label in target is smaller than it).

    When ``num_classes`` is not specified in these cases, consistency of the highest target
    value against ``C`` dimension is checked for (multi-dimensional) multi-class cases.

    If ``top_k`` is set (not None) for inputs that do not have probability predictions (and
    are not binary), an error is raised. Similarly, if ``top_k`` is set to a number that
    is higher than or equal to the ``C`` dimension of ``preds``, an error is raised.

    Preds and target tensors are expected to be squeezed already - all dimensions should be
    greater than 1, except perhaps the first one (``N``).

    Args:
        preds: Tensor with predictions (labels or probabilities)
        target: Tensor with ground truth labels, always integers (labels)
        threshold:
            Threshold value for transforming probability/logit predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs.
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be inferred
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of the highest probability entries for each sample to convert to 1s - relevant
            only for inputs with probability predictions. The default value (``None``) will be
            interpreted as 1 for these inputs. If this parameter is set for multi-label inputs,
            it will take precedence over threshold.

            Should be left unset (``None``) for inputs with label predictions.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/overview:using the multiclass parameter>`
            for a more detailed explanation and examples.
        ignore_index: ignore predictions where targets are equal to this number


    Return:
        case: The case the inputs fall in, one of 'binary', 'multi-class', 'multi-label' or
            'multi-dim multi-class'

    """
    _basic_input_validation(preds, target, threshold, multiclass, ignore_index)
    case, implied_classes = _check_shape_and_type_consistency(preds, target)
    if preds.shape != target.shape:
        if multiclass is False and implied_classes != 2:
            raise ValueError('You have set `multiclass=False`, but have more than 2 classes in your data, based on the C dimension of `preds`.')
        if target.max() >= implied_classes:
            raise ValueError('The highest label in `target` should be smaller than the size of the `C` dimension of `preds`.')
    if num_classes:
        if case == DataType.BINARY:
            _check_num_classes_binary(num_classes, multiclass)
        elif case in (DataType.MULTICLASS, DataType.MULTIDIM_MULTICLASS):
            _check_num_classes_mc(preds, target, num_classes, multiclass, implied_classes)
        elif case.MULTILABEL:
            _check_num_classes_ml(num_classes, multiclass, implied_classes)
    if top_k is not None:
        _check_top_k(top_k, case, implied_classes, multiclass, preds.is_floating_point())
    return case