from typing import Any, List, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.calibration_error import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class BinaryCalibrationError(Metric):
    """`Top-label Calibration Error`_ for binary tasks.

    The expected calibration error can be used to quantify how well a given model is calibrated e.g. how well the
    predicted output probabilities of the model matches the actual probabilities of the ground truth distribution.
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    .. math::
        \\text{ECE} = \\sum_i^N b_i \\|(p_i - c_i)\\|, \\text{L1 norm (Expected Calibration Error)}

    .. math::
        \\text{MCE} =  \\max_{i} (p_i - c_i), \\text{Infinity norm (Maximum Calibration Error)}

    .. math::
        \\text{RMSCE} = \\sqrt{\\sum_i^N b_i(p_i - c_i)^2}, \\text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    in an uniform way in the [0,1] range.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)`` containing probabilities or logits for
      each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the
      positive class.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bce`` (:class:`~torch.Tensor`): A scalar tensor containing the calibration error

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification import BinaryCalibrationError
        >>> preds = tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = tensor([0, 0, 1, 1, 1])
        >>> metric = BinaryCalibrationError(n_bins=2, norm='l1')
        >>> metric(preds, target)
        tensor(0.2900)
        >>> bce = BinaryCalibrationError(n_bins=2, norm='l2')
        >>> bce(preds, target)
        tensor(0.2918)
        >>> bce = BinaryCalibrationError(n_bins=2, norm='max')
        >>> bce(preds, target)
        tensor(0.3167)

    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    confidences: List[Tensor]
    accuracies: List[Tensor]

    def __init__(self, n_bins: int=15, norm: Literal['l1', 'l2', 'max']='l1', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        self.validate_args = validate_args
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.add_state('confidences', [], dist_reduce_fx='cat')
        self.add_state('accuracies', [], dist_reduce_fx='cat')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _binary_calibration_error_tensor_validation(preds, target, self.ignore_index)
        preds, target = _binary_confusion_matrix_format(preds, target, threshold=0.0, ignore_index=self.ignore_index, convert_to_labels=False)
        confidences, accuracies = _binary_calibration_error_update(preds, target)
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)

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

            >>> from torch import rand, randint
            >>> # Example plotting a single value
            >>> from torchmetrics.classification import BinaryCalibrationError
            >>> metric = BinaryCalibrationError(n_bins=2, norm='l1')
            >>> metric.update(rand(10), randint(2,(10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import rand, randint
            >>> # Example plotting multiple values
            >>> from torchmetrics.classification import BinaryCalibrationError
            >>> metric = BinaryCalibrationError(n_bins=2, norm='l1')
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2,(10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)