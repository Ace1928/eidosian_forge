from typing import Any, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.confusion_matrix import BinaryConfusionMatrix, MulticlassConfusionMatrix
from torchmetrics.functional.classification.cohen_kappa import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class CohenKappa(_ClassificationTaskWrapper):
    """Calculate `Cohen's kappa score`_ that measures inter-annotator agreement.

    .. math::
        \\kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :class:`~torchmetrics.classification.BinaryCohenKappa` and
    :class:`~torchmetrics.classification.MulticlassCohenKappa` for the specific details of each argument influence and
    examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> cohenkappa = CohenKappa(task="multiclass", num_classes=2)
        >>> cohenkappa(preds, target)
        tensor(0.5000)

    """

    def __new__(cls: Type['CohenKappa'], task: Literal['binary', 'multiclass'], threshold: float=0.5, num_classes: Optional[int]=None, weights: Optional[Literal['linear', 'quadratic', 'none']]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update({'weights': weights, 'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCohenKappa(threshold, **kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            return MulticlassCohenKappa(num_classes, **kwargs)
        raise ValueError(f'Task {task} not supported!')