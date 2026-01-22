from typing import Any, Optional, Sequence, Union
from torch import Tensor, tensor
from torchmetrics.functional.audio.snr import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class ComplexScaleInvariantSignalNoiseRatio(Metric):
    """Calculate `Complex scale-invariant signal-to-noise ratio`_ (C-SI-SNR) metric for evaluating quality of audio.

    As input to `forward` and `update` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): real float tensor with shape ``(...,frequency,time,2)`` or complex float
      tensor with shape ``(..., frequency,time)``

    - ``target`` (:class:`~torch.Tensor`): real float tensor with shape ``(...,frequency,time,2)`` or complex float
      tensor with shape ``(..., frequency,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``c_si_snr`` (:class:`~torch.Tensor`): float scalar tensor with average C-SI-SNR value over samples

    Args:
        zero_mean: if to zero mean target and preds or not
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``zero_mean`` is not an bool
        TypeError:
            If ``preds`` is not the shape (..., frequency, time, 2) (after being converted to real if it is complex).
            If ``preds`` and ``target`` does not have the same shape.

    Example:
        >>> import torch
        >>> from torch import tensor
        >>> from torchmetrics.audio import ComplexScaleInvariantSignalNoiseRatio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn((1,257,100,2))
        >>> target = torch.randn((1,257,100,2))
        >>> c_si_snr = ComplexScaleInvariantSignalNoiseRatio()
        >>> c_si_snr(preds, target)
        tensor(-63.4849)

    """
    is_differentiable = True
    ci_snr_sum: Tensor
    num: Tensor
    higher_is_better = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, zero_mean: bool=False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(zero_mean, bool):
            raise ValueError(f'Expected argument `zero_mean` to be an bool, but got {zero_mean}')
        self.zero_mean = zero_mean
        self.add_state('ci_snr_sum', default=tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num', default=tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        v = complex_scale_invariant_signal_noise_ratio(preds=preds, target=target, zero_mean=self.zero_mean)
        self.ci_snr_sum += v.sum()
        self.num += v.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.ci_snr_sum / self.num

    def plot(self, val: Union[Tensor, Sequence[Tensor], None]=None, ax: Optional[_AX_TYPE]=None) -> _PLOT_OUT_TYPE:
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

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.audio import ComplexScaleInvariantSignalNoiseRatio
            >>> metric = ComplexScaleInvariantSignalNoiseRatio()
            >>> metric.update(torch.rand(1,257,100,2), torch.rand(1,257,100,2))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import ComplexScaleInvariantSignalNoiseRatio
            >>> metric = ComplexScaleInvariantSignalNoiseRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(1,257,100,2), torch.rand(1,257,100,2)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)