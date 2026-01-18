from typing import Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_xlogy
Compute `KL divergence`_.

    .. math::
        D_{KL}(P||Q) = \sum_{x\in\mathcal{X}} P(x) \log\frac{P(x)}{Q{x}}

    Where :math:`P` and :math:`Q` are probability distributions where :math:`P` usually represents a distribution
    over data and :math:`Q` is often a prior or approximation of :math:`P`. It should be noted that the KL divergence
    is a non-symmetrical metric i.e. :math:`D_{KL}(P||Q) \neq D_{KL}(Q||P)`.

    Args:
        p: data distribution with shape ``[N, d]``
        q: prior or approximate distribution with shape ``[N, d]``
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributes sum to 1
        reduction:
            Determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

    Example:
        >>> from torch import tensor
        >>> p = tensor([[0.36, 0.48, 0.16]])
        >>> q = tensor([[1/3, 1/3, 1/3]])
        >>> kl_divergence(p, q)
        tensor(0.0853)

    