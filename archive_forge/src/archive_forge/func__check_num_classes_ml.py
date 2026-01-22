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
def _check_num_classes_ml(num_classes: int, multiclass: Optional[bool], implied_classes: int) -> None:
    """Check that the consistency of ``num_classes`` with the data and ``multiclass`` param for multi-label data."""
    if multiclass and num_classes != 2:
        raise ValueError('Your have set `multiclass=True`, but `num_classes` is not equal to 2. If you are trying to transform multi-label data to 2 class multi-dimensional multi-class, you should set `num_classes` to either 2 or None.')
    if not multiclass and num_classes != implied_classes:
        raise ValueError('The implied number of classes (from shape of inputs) does not match num_classes.')