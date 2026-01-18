from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTask
Calculate `Matthews correlation coefficient`_ .

    This metric measures the general correlation or quality of a classification.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_matthews_corrcoef`,
    :func:`~torchmetrics.functional.classification.multiclass_matthews_corrcoef` and
    :func:`~torchmetrics.functional.classification.multilabel_matthews_corrcoef` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> matthews_corrcoef(preds, target, task="multiclass", num_classes=2)
        tensor(0.5774)

    