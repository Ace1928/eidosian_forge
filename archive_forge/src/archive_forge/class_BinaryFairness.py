from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.group_fairness import (
from torchmetrics.functional.classification.stat_scores import _binary_stat_scores_arg_validation
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class BinaryFairness(_AbstractGroupStatScores):
    """Computes `Demographic parity`_ and `Equal opportunity`_ ratio for binary classification problems.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``groups`` (int tensor): ``(N, ...)``. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
    - ``target`` (int tensor): ``(N, ...)``.

    The additional dimensions are flatted along the batch dimension.

    This class computes the ratio between positivity rates and true positives rates for different groups.
    If more than two groups are present, the disparity between the lowest and highest group is reported.
    A disparity between positivity rates indicates a potential violation of demographic parity, and between
    true positive rates indicates a potential violation of equal opportunity.

    The lowest rate is divided by the highest, so a lower value means more discrimination against the numerator.
    In the results this is also indicated as the key of dict is {metric}_{identifier_low_group}_{identifier_high_group}.

    Args:
        num_groups: The number of groups.
        task: The task to compute. Can be either ``demographic_parity`` or ``equal_oppotunity`` or ``all``.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        The metric returns a dict where the key identifies the metric and groups with the lowest and highest true
        positives rates as follows: {metric}__{identifier_low_group}_{identifier_high_group}.
        The value is a tensor with the disparity rate.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import BinaryFairness
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> groups = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> metric = BinaryFairness(2)
        >>> metric(preds, target, groups)
        {'DP_0_1': tensor(0.), 'EO_0_1': tensor(0.)}

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryFairness
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0.11, 0.84, 0.22, 0.73, 0.33, 0.92])
        >>> groups = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> metric = BinaryFairness(2)
        >>> metric(preds, target, groups)
        {'DP_0_1': tensor(0.), 'EO_0_1': tensor(0.)}

    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(self, num_groups: int, task: Literal['demographic_parity', 'equal_opportunity', 'all']='all', threshold: float=0.5, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__()
        if task not in ['demographic_parity', 'equal_opportunity', 'all']:
            raise ValueError(f'Expected argument `task` to either be ``demographic_parity``,``equal_opportunity`` or ``all`` but got {task}.')
        if validate_args:
            _binary_stat_scores_arg_validation(threshold, 'global', ignore_index)
        if not isinstance(num_groups, int) and num_groups < 2:
            raise ValueError(f'Expected argument `num_groups` to be an int larger than 1, but got {num_groups}')
        self.num_groups = num_groups
        self.task = task
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self._create_states(self.num_groups)

    def update(self, preds: Tensor, target: Tensor, groups: Tensor) -> None:
        """Update state with predictions, groups, and target.

        Args:
            preds: Tensor with predictions.
            target: Tensor with true labels.
            groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

        """
        if self.task == 'demographic_parity':
            if target is not None:
                rank_zero_warn('The task demographic_parity does not require a target.', UserWarning)
            target = torch.zeros(preds.shape)
        group_stats = _binary_groups_stat_scores(preds, target, groups, self.num_groups, self.threshold, self.ignore_index, self.validate_args)
        self._update_states(group_stats)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute fairness criteria based on inputs passed in to ``update`` previously."""
        if self.task == 'demographic_parity':
            return _compute_binary_demographic_parity(self.tp, self.fp, self.tn, self.fn)
        if self.task == 'equal_opportunity':
            return _compute_binary_equal_opportunity(self.tp, self.fp, self.tn, self.fn)
        if self.task == 'all':
            return {**_compute_binary_demographic_parity(self.tp, self.fp, self.tn, self.fn), **_compute_binary_equal_opportunity(self.tp, self.fp, self.tn, self.fn)}
        return None

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

            >>> import torch
            >>> _ = torch.manual_seed(42)
            >>> # Example plotting a single value
            >>> from torchmetrics.classification import BinaryFairness
            >>> metric = BinaryFairness(2)
            >>> metric.update(torch.rand(20), torch.randint(2,(20,)), torch.randint(2,(20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> import torch
            >>> _ = torch.manual_seed(42)
            >>> # Example plotting multiple values
            >>> from torchmetrics.classification import BinaryFairness
            >>> metric = BinaryFairness(2)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(20), torch.randint(2,(20,)), torch.ones(20).long()))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)