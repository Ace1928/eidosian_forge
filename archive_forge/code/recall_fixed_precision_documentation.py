from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.enums import ClassificationTask
Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for a
    given precision level.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_recall_at_fixed_precision`,
    :func:`~torchmetrics.functional.classification.multiclass_recall_at_fixed_precision` and
    :func:`~torchmetrics.functional.classification.multilabel_recall_at_fixed_precision` for the specific details of
    each argument influence and examples.

    