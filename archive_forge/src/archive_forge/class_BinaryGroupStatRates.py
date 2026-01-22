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
class BinaryGroupStatRates(_AbstractGroupStatScores):
    """Computes the true/false positives and true/false negatives rates for binary classification by group.

    Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``.
    - ``groups`` (int tensor): ``(N, ...)``. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

    The additional dimensions are flatted along the batch dimension.

    Args:
        num_groups: The number of groups.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        The metric returns a dict with a group identifier as key and a tensor with the tp, fp, tn and fn rates as value.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import BinaryGroupStatRates
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> groups = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> metric = BinaryGroupStatRates(num_groups=2)
        >>> metric(preds, target, groups)
        {'group_0': tensor([0., 0., 1., 0.]), 'group_1': tensor([1., 0., 0., 0.])}

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryGroupStatRates
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0.11, 0.84, 0.22, 0.73, 0.33, 0.92])
        >>> groups = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> metric = BinaryGroupStatRates(num_groups=2)
        >>> metric(preds, target, groups)
        {'group_0': tensor([0., 0., 1., 0.]), 'group_1': tensor([1., 0., 0., 0.])}

    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(self, num_groups: int, threshold: float=0.5, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__()
        if validate_args:
            _binary_stat_scores_arg_validation(threshold, 'global', ignore_index)
        if not isinstance(num_groups, int) and num_groups < 2:
            raise ValueError(f'Expected argument `num_groups` to be an int larger than 1, but got {num_groups}')
        self.num_groups = num_groups
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self._create_states(self.num_groups)

    def update(self, preds: Tensor, target: Tensor, groups: Tensor) -> None:
        """Update state with predictions, target and group identifiers.

        Args:
            preds: Tensor with predictions.
            target: Tensor with true labels.
            groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

        """
        group_stats = _binary_groups_stat_scores(preds, target, groups, self.num_groups, self.threshold, self.ignore_index, self.validate_args)
        self._update_states(group_stats)

    def compute(self) -> Dict[str, Tensor]:
        """Compute tp, fp, tn and fn rates based on inputs passed in to ``update`` previously."""
        results = torch.stack((self.tp, self.fp, self.tn, self.fn), dim=1)
        return {f'group_{i}': group / group.sum() for i, group in enumerate(results)}