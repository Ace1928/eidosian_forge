from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
Calculate the Jaccard index.

    The `Jaccard index`_ (also known as the intersetion over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_jaccard_index`,
    :func:`~torchmetrics.functional.classification.multiclass_jaccard_index` and
    :func:`~torchmetrics.functional.classification.multilabel_jaccard_index` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> from torch import randint, tensor
        >>> target = randint(0, 2, (10, 25, 25))
        >>> pred = tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> jaccard_index(pred, target, task="multiclass", num_classes=2)
        tensor(0.9660)

    