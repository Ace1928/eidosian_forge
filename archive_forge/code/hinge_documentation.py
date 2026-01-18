from typing import Optional, Tuple
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs).

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_hinge_loss` and
    :func:`~torchmetrics.functional.classification.multiclass_hinge_loss` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([0, 1, 1])
        >>> preds = tensor([0.5, 0.7, 0.1])
        >>> hinge_loss(preds, target, task="binary")
        tensor(0.9000)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge_loss(preds, target, task="multiclass", num_classes=3)
        tensor(1.5551)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge_loss(preds, target, task="multiclass", num_classes=3, multiclass_mode="one-vs-all")
        tensor([1.3743, 1.1945, 1.2359])

    