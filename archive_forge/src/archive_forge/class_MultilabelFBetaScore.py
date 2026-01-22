from typing import Any, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.stat_scores import BinaryStatScores, MulticlassStatScores, MultilabelStatScores
from torchmetrics.functional.classification.f_beta import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class MultilabelFBetaScore(MultilabelStatScores):
    """Compute `F-score`_ metric for multilabel tasks.

    .. math::
        F_{\\beta} = (1 + \\beta^2) * \\frac{\\text{precision} * \\text{recall}}
        {(\\beta^2 * \\text{precision}) + \\text{recall}}

    The metric is only proper defined when :math:`\\text{TP} + \\text{FP} \\neq 0 \\wedge \\text{TP} + \\text{FN} \\neq 0`
    where :math:`\\text{TP}`, :math:`\\text{FP}` and :math:`\\text{FN}` represent the number of true positives, false
    positives and false negatives respectively. If this case is encountered for any label, the metric for that label
    will be set to 0 and the overall metric may therefore be affected in turn.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int or float tensor of shape ``(N, C, ...)``. If preds is a floating
      point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid
      per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlfbs`` (:class:`~torch.Tensor`): A tensor whose returned shape depends on the ``average`` and
      ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MultilabelFBetaScore
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelFBetaScore(beta=2.0, num_labels=3)
        >>> metric(preds, target)
        tensor(0.6111)
        >>> mlfbs = MultilabelFBetaScore(beta=2.0, num_labels=3, average=None)
        >>> mlfbs(preds, target)
        tensor([1.0000, 0.0000, 0.8333])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelFBetaScore
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelFBetaScore(beta=2.0, num_labels=3)
        >>> metric(preds, target)
        tensor(0.6111)
        >>> mlfbs = MultilabelFBetaScore(beta=2.0, num_labels=3, average=None)
        >>> mlfbs(preds, target)
        tensor([1.0000, 0.0000, 0.8333])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MultilabelFBetaScore
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99],  [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = MultilabelFBetaScore(num_labels=3, beta=2.0, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.5556, 0.0000])
        >>> mlfbs = MultilabelFBetaScore(num_labels=3, beta=2.0, multidim_average='samplewise', average=None)
        >>> mlfbs(preds, target)
        tensor([[0.8333, 0.8333, 0.0000],
                [0.0000, 0.0000, 0.0000]])

    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = 'Label'

    def __init__(self, beta: float, num_labels: int, threshold: float=0.5, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(num_labels=num_labels, threshold=threshold, average=average, multidim_average=multidim_average, ignore_index=ignore_index, validate_args=False, **kwargs)
        if validate_args:
            _multilabel_fbeta_score_arg_validation(beta, num_labels, threshold, average, multidim_average, ignore_index)
        self.validate_args = validate_args
        self.beta = beta

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _fbeta_reduce(tp, fp, tn, fn, self.beta, average=self.average, multidim_average=self.multidim_average, multilabel=True)

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

            >>> from torch import rand, randint
            >>> # Example plotting a single value
            >>> from torchmetrics.classification import MultilabelFBetaScore
            >>> metric = MultilabelFBetaScore(num_labels=3, beta=2.0)
            >>> metric.update(randint(2, (20, 3)), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import rand, randint
            >>> # Example plotting multiple values
            >>> from torchmetrics.classification import MultilabelFBetaScore
            >>> metric = MultilabelFBetaScore(num_labels=3, beta=2.0)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3)), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)