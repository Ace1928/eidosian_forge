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
class MultilabelHammingDistance(MultilabelStatScores):
    """Compute the average `Hamming distance`_ (also known as Hamming loss) for multilabel tasks.

    .. math::
        \\text{Hamming distance} = \\frac{1}{N \\cdot L} \\sum_i^N \\sum_l^L 1(y_{il} \\neq \\hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\\hat{y}` is a tensor of predictions,
    and :math:`\\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor or float tensor of shape ``(N, C, ...)``. If preds is a
      floating point tensor with values outside [0,1] range we consider the input to be logits and will auto
      apply sigmoid per element. Additionally, we convert to int tensor with thresholding using the value in
      ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlhd`` (:class:`~torch.Tensor`): A tensor whose returned shape depends on the ``average`` and
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
        >>> from torchmetrics.classification import MultilabelHammingDistance
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelHammingDistance(num_labels=3)
        >>> metric(preds, target)
        tensor(0.3333)
        >>> mlhd = MultilabelHammingDistance(num_labels=3, average=None)
        >>> mlhd(preds, target)
        tensor([0.0000, 0.5000, 0.5000])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelHammingDistance
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelHammingDistance(num_labels=3)
        >>> metric(preds, target)
        tensor(0.3333)
        >>> mlhd = MultilabelHammingDistance(num_labels=3, average=None)
        >>> mlhd(preds, target)
        tensor([0.0000, 0.5000, 0.5000])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MultilabelHammingDistance
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = MultilabelHammingDistance(num_labels=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.6667, 0.8333])
        >>> mlhd = MultilabelHammingDistance(num_labels=3, multidim_average='samplewise', average=None)
        >>> mlhd(preds, target)
        tensor([[0.5000, 0.5000, 1.0000],
                [1.0000, 1.0000, 0.5000]])

    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = 'Label'

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _hamming_distance_reduce(tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average, multilabel=True)

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

            >>> # Example plotting a single value
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import MultilabelHammingDistance
            >>> metric = MultilabelHammingDistance(num_labels=3)
            >>> metric.update(randint(2, (20, 3)), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import MultilabelHammingDistance
            >>> metric = MultilabelHammingDistance(num_labels=3)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3)), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)