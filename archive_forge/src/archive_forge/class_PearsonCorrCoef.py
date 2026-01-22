from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute, _pearson_corrcoef_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class PearsonCorrCoef(Metric):
    """Compute `Pearson Correlation Coefficient`_.

    .. math::
        P_{corr}(x,y) = \\frac{cov(x,y)}{\\sigma_x \\sigma_y}

    Where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): either single output float tensor with shape ``(N,)``
      or multioutput float tensor of shape ``(N,d)``
    - ``target`` (:class:`~torch.Tensor`): either single output tensor with shape ``(N,)``
      or multioutput tensor of shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``pearson`` (:class:`~torch.Tensor`): A tensor with the Pearson Correlation Coefficient

    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (single output regression):
        >>> from torchmetrics.regression import PearsonCorrCoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson = PearsonCorrCoef()
        >>> pearson(preds, target)
        tensor(0.9849)

    Example (multi output regression):
        >>> from torchmetrics.regression import PearsonCorrCoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> pearson = PearsonCorrCoef(num_outputs=2)
        >>> pearson(preds, target)
        tensor([1., 1.])

    """
    is_differentiable: bool = True
    higher_is_better: Optional[bool] = None
    full_state_update: bool = True
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]
    mean_x: Tensor
    mean_y: Tensor
    var_x: Tensor
    var_y: Tensor
    corr_xy: Tensor
    n_total: Tensor

    def __init__(self, num_outputs: int=1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError('Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}')
        self.num_outputs = num_outputs
        self.add_state('mean_x', default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state('mean_y', default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state('var_x', default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state('var_y', default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state('corr_xy', default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state('n_total', default=torch.zeros(self.num_outputs), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total = _pearson_corrcoef_update(preds, target, self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total, self.num_outputs)

    def compute(self) -> Tensor:
        """Compute pearson correlation coefficient over state."""
        if self.num_outputs == 1 and self.mean_x.numel() > 1 or (self.num_outputs > 1 and self.mean_x.ndim > 1):
            _, _, var_x, var_y, corr_xy, n_total = _final_aggregation(self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total)
        else:
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total
        return _pearson_corrcoef_compute(var_x, var_y, corr_xy, n_total)

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]]=None, ax: Optional[_AX_TYPE]=None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting a single value
            >>> from torchmetrics.regression import PearsonCorrCoef
            >>> metric = PearsonCorrCoef()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import PearsonCorrCoef
            >>> metric = PearsonCorrCoef()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)