from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
class StatScores(_ClassificationTaskWrapper):
    """Compute the number of true positives, false positives, true negatives, false negatives and the support.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :class:`~torchmetrics.classification.BinaryStatScores`, :class:`~torchmetrics.classification.MulticlassStatScores`
    and :class:`~torchmetrics.classification.MultilabelStatScores` for the specific details of each argument influence
    and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> preds  = tensor([1, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> stat_scores = StatScores(task="multiclass", num_classes=3, average='micro')
        >>> stat_scores(preds, target)
        tensor([2, 2, 6, 2, 4])
        >>> stat_scores = StatScores(task="multiclass", num_classes=3, average=None)
        >>> stat_scores(preds, target)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])

    """

    def __new__(cls: Type['StatScores'], task: Literal['binary', 'multiclass', 'multilabel'], threshold: float=0.5, num_classes: Optional[int]=None, num_labels: Optional[int]=None, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='micro', multidim_average: Optional[Literal['global', 'samplewise']]='global', top_k: Optional[int]=1, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        assert multidim_average is not None
        kwargs.update({'multidim_average': multidim_average, 'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryStatScores(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            if not isinstance(top_k, int):
                raise ValueError(f'`top_k` is expected to be `int` but `{type(top_k)} was passed.`')
            return MulticlassStatScores(num_classes, top_k, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
            return MultilabelStatScores(num_labels, threshold, average, **kwargs)
        raise ValueError(f'Task {task} not supported!')