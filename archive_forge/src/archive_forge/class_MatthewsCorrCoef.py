from typing import Any, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.confusion_matrix import (
from torchmetrics.functional.classification.matthews_corrcoef import _matthews_corrcoef_reduce
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class MatthewsCorrCoef(_ClassificationTaskWrapper):
    """Calculate `Matthews correlation coefficient`_ .

    This metric measures the general correlation or quality of a classification.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :class:`~torchmetrics.classification.BinaryMatthewsCorrCoef`,
    :class:`~torchmetrics.classification.MulticlassMatthewsCorrCoef` and
    :class:`~torchmetrics.classification.MultilabelMatthewsCorrCoef` for the specific details of each argument influence
    and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> matthews_corrcoef = MatthewsCorrCoef(task='binary')
        >>> matthews_corrcoef(preds, target)
        tensor(0.5774)

    """

    def __new__(cls: Type['MatthewsCorrCoef'], task: Literal['binary', 'multiclass', 'multilabel'], threshold: float=0.5, num_classes: Optional[int]=None, num_labels: Optional[int]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryMatthewsCorrCoef(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            return MulticlassMatthewsCorrCoef(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
            return MultilabelMatthewsCorrCoef(num_labels, threshold, **kwargs)
        raise ValueError(f'Not handled value: {task}')