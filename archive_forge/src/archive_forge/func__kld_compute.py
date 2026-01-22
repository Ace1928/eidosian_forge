from typing import Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_xlogy
def _kld_compute(measures: Tensor, total: Union[int, Tensor], reduction: Literal['mean', 'sum', 'none', None]='mean') -> Tensor:
    """Compute the KL divergenece based on the type of reduction.

    Args:
        measures: Tensor of KL divergence scores for each observation
        total: Number of observations
        reduction:
            Determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

    Example:
        >>> p = torch.tensor([[0.36, 0.48, 0.16]])
        >>> q = torch.tensor([[1/3, 1/3, 1/3]])
        >>> measures, total = _kld_update(p, q, log_prob=False)
        >>> _kld_compute(measures, total)
        tensor(0.0853)

    """
    if reduction == 'sum':
        return measures.sum()
    if reduction == 'mean':
        return measures.sum() / total
    if reduction is None or reduction == 'none':
        return measures
    return measures / total