from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
Compute the average `Hamming distance`_ (also known as Hamming loss).

    .. math::
        \text{Hamming distance} = \frac{1}{N \cdot L} \sum_i^N \sum_l^L 1(y_{il} \neq \hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions,
    and :math:`\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_hamming_distance`,
    :func:`~torchmetrics.functional.classification.multiclass_hamming_distance` and
    :func:`~torchmetrics.functional.classification.multilabel_hamming_distance` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([[0, 1], [1, 1]])
        >>> preds = tensor([[0, 1], [0, 1]])
        >>> hamming_distance(preds, target, task="binary")
        tensor(0.2500)

    