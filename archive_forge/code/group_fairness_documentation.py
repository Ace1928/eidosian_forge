from typing import Dict, List, Optional, Tuple
import torch
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _flexible_bincount
Compute either `Demographic parity`_ and `Equal opportunity`_ ratio for binary classification problems.

    This is done by setting the ``task`` argument to either ``'demographic_parity'``, ``'equal_opportunity'``
    or ``all``. See the documentation of
    :func:`~torchmetrics.functional.classification.demographic_parity`
    and :func:`~torchmetrics.functional.classification.equal_opportunity` for the specific details of
    each argument influence and examples.

    Args:
        preds: Tensor with predictions.
        target: Tensor with true labels (not required for demographic_parity).
        groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
        task: The task to compute. Can be either ``demographic_parity`` or ``equal_oppotunity`` or ``all``.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    