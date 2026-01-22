from typing import Optional, Tuple
from torch import Tensor
def _reduce_distance_matrix(distmat: Tensor, reduction: Optional[str]=None) -> Tensor:
    """Reduction of distance matrix.

    Args:
        distmat: a ``[N,M]`` matrix
        reduction: string determining how to reduce along last dimension

    """
    if reduction == 'mean':
        return distmat.mean(dim=-1)
    if reduction == 'sum':
        return distmat.sum(dim=-1)
    if reduction is None or reduction == 'none':
        return distmat
    raise ValueError(f"Expected reduction to be one of `['mean', 'sum', None]` but got {reduction}")