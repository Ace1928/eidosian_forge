from typing import Any, Optional, Sequence, Type, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.exact_match import (
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoBinary
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class MulticlassExactMatch(Metric):
    """Compute Exact match (also known as subset accuracy) for multiclass tasks.

    Exact Match is a stricter version of accuracy where all labels have to match exactly for the sample to be
    correctly classified.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcem`` (:class:`~torch.Tensor`): A tensor whose returned shape depends on the ``multidim_average`` argument:

        - If ``multidim_average`` is set to ``global`` the output will be a scalar tensor
        - If ``multidim_average`` is set to ``samplewise`` the output will be a tensor of shape ``(N,)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        num_classes: Integer specifying the number of labels
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (multidim tensors):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassExactMatch
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassExactMatch(num_classes=3, multidim_average='global')
        >>> metric(preds, target)
        tensor(0.5000)

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassExactMatch
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassExactMatch(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([1., 0.])

    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = 'Class'

    def __init__(self, num_classes: int, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        top_k, average = (1, None)
        if validate_args:
            _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        self.num_classes = num_classes
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.add_state('correct', torch.zeros(1, dtype=torch.long) if self.multidim_average == 'global' else [], dist_reduce_fx='sum' if self.multidim_average == 'global' else 'cat')
        self.add_state('total', torch.zeros(1, dtype=torch.long), dist_reduce_fx='sum' if self.multidim_average == 'global' else 'mean')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(preds, target, self.num_classes, self.multidim_average, self.ignore_index)
        preds, target = _multiclass_stat_scores_format(preds, target, 1)
        correct, total = _multiclass_exact_match_update(preds, target, self.multidim_average, self.ignore_index)
        if self.multidim_average == 'samplewise':
            self.correct.append(correct)
            self.total = total
        else:
            self.correct += correct
            self.total += total

    def compute(self) -> Tensor:
        """Compute metric."""
        correct = dim_zero_cat(self.correct) if isinstance(self.correct, list) else self.correct
        return _exact_match_reduce(correct, self.total)

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]]=None, ax: Optional[_AX_TYPE]=None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value per class
            >>> from torch import randint
            >>> from torchmetrics.classification import MulticlassExactMatch
            >>> metric = MulticlassExactMatch(num_classes=3)
            >>> metric.update(randint(3, (20,5)), randint(3, (20,5)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randint
            >>> # Example plotting a multiple values per class
            >>> from torchmetrics.classification import MulticlassExactMatch
            >>> metric = MulticlassExactMatch(num_classes=3)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20,5)), randint(3, (20,5))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)