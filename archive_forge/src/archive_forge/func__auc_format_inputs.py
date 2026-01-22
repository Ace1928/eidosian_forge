from typing import Optional, Tuple
import torch
from torch import Tensor
def _auc_format_inputs(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """Check that auc input is correct."""
    x = x.squeeze() if x.ndim > 1 else x
    y = y.squeeze() if y.ndim > 1 else y
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(f'Expected both `x` and `y` tensor to be 1d, but got tensors with dimension {x.ndim} and {y.ndim}')
    if x.numel() != y.numel():
        raise ValueError(f'Expected the same number of elements in `x` and `y` tensor but received {x.numel()} and {y.numel()}')
    return (x, y)