from typing import Any, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.stat_scores import BinaryStatScores, MulticlassStatScores, MultilabelStatScores
from torchmetrics.functional.classification.hamming import _hamming_distance_reduce
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class HammingDistance(_ClassificationTaskWrapper):
    """Compute the average `Hamming distance`_ (also known as Hamming loss).

    .. math::
        \\text{Hamming distance} = \\frac{1}{N \\cdot L} \\sum_i^N \\sum_l^L 1(y_{il} \\neq \\hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\\hat{y}` is a tensor of predictions,
    and :math:`\\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :class:`~torchmetrics.classification.BinaryHammingDistance`,
    :class:`~torchmetrics.classification.MulticlassHammingDistance` and
    :class:`~torchmetrics.classification.MultilabelHammingDistance` for the specific details of each argument influence
    and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([[0, 1], [1, 1]])
        >>> preds = tensor([[0, 1], [0, 1]])
        >>> hamming_distance = HammingDistance(task="multilabel", num_labels=2)
        >>> hamming_distance(preds, target)
        tensor(0.2500)

    """

    def __new__(cls: Type['HammingDistance'], task: Literal['binary', 'multiclass', 'multilabel'], threshold: float=0.5, num_classes: Optional[int]=None, num_labels: Optional[int]=None, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='micro', multidim_average: Optional[Literal['global', 'samplewise']]='global', top_k: Optional[int]=1, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        assert multidim_average is not None
        kwargs.update({'multidim_average': multidim_average, 'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryHammingDistance(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            if not isinstance(top_k, int):
                raise ValueError(f'`top_k` is expected to be `int` but `{type(top_k)} was passed.`')
            return MulticlassHammingDistance(num_classes, top_k, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
            return MultilabelHammingDistance(num_labels, threshold, average, **kwargs)
        raise ValueError(f'Task {task} not supported!')