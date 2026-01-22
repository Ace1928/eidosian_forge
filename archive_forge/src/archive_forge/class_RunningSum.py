from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.running import Running
class RunningSum(Running):
    """Aggregate a stream of value into their sum over a running window.

    Using this metric compared to `SumMetric` allows for calculating metrics over a running window of values, instead
    of the whole history of values. This is beneficial when you want to get a better estimate of the metric during
    training and don't want to wait for the whole training to finish to get epoch level estimates.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated sum over all inputs received

    Args:
        window: The size of the running window.
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.aggregation import RunningSum
        >>> metric = RunningSum(window=3)
        >>> for i in range(6):
        ...     current_val = metric(tensor([i]))
        ...     running_val = metric.compute()
        ...     total_val = tensor(sum(list(range(i+1))))  # total sum over all samples
        ...     print(f"{current_val=}, {running_val=}, {total_val=}")
        current_val=tensor(0.), running_val=tensor(0.), total_val=tensor(0)
        current_val=tensor(1.), running_val=tensor(1.), total_val=tensor(1)
        current_val=tensor(2.), running_val=tensor(3.), total_val=tensor(3)
        current_val=tensor(3.), running_val=tensor(6.), total_val=tensor(6)
        current_val=tensor(4.), running_val=tensor(9.), total_val=tensor(10)
        current_val=tensor(5.), running_val=tensor(12.), total_val=tensor(15)

    """

    def __init__(self, window: int=5, nan_strategy: Union[str, float]='warn', **kwargs: Any) -> None:
        super().__init__(base_metric=SumMetric(nan_strategy=nan_strategy, **kwargs), window=window)