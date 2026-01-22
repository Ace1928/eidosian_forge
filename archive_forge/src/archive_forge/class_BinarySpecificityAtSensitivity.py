from typing import Any, List, Optional, Tuple, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.precision_recall_curve import (
from torchmetrics.functional.classification.specificity_sensitivity import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat as _cat
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
class BinarySpecificityAtSensitivity(BinaryPrecisionRecallCurve):
    """Compute the highest possible specificity value given the minimum sensitivity thresholds provided.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and the
    find the specificity for a given sensitivity level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        min_sensitivity: float value specifying minimum sensitivity threshold.
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        (tuple): a tuple of 2 tensors containing:

        - specificity: an scalar tensor with the maximum specificity for the given sensitivity level
        - threshold: an scalar tensor with the corresponding threshold level

    Example:
        >>> from torchmetrics.classification import BinarySpecificityAtSensitivity
        >>> from torch import tensor
        >>> preds = tensor([0, 0.5, 0.4, 0.1])
        >>> target = tensor([0, 1, 1, 1])
        >>> metric = BinarySpecificityAtSensitivity(min_sensitivity=0.5, thresholds=None)
        >>> metric(preds, target)
        (tensor(1.), tensor(0.4000))
        >>> metric = BinarySpecificityAtSensitivity(min_sensitivity=0.5, thresholds=5)
        >>> metric(preds, target)
        (tensor(1.), tensor(0.2500))

    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(self, min_sensitivity: float, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(thresholds, ignore_index, validate_args=False, **kwargs)
        if validate_args:
            _binary_specificity_at_sensitivity_arg_validation(min_sensitivity, thresholds, ignore_index)
        self.validate_args = validate_args
        self.min_sensitivity = min_sensitivity

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Compute metric."""
        state = (_cat(self.preds), _cat(self.target)) if self.thresholds is None else self.confmat
        return _binary_specificity_at_sensitivity_compute(state, self.thresholds, self.min_sensitivity)