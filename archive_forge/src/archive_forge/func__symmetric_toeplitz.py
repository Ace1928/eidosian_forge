import math
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.linalg import norm
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _FAST_BSS_EVAL_AVAILABLE
def _symmetric_toeplitz(vector: Tensor) -> Tensor:
    """Construct a symmetric Toeplitz matrix using one vector.

    Args:
        vector: shape [..., L]

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.functional.audio.sdr import _symmetric_toeplitz
        >>> v = tensor([0, 1, 2, 3, 4])
        >>> _symmetric_toeplitz(v)
        tensor([[0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0]])

    Returns:
        a symmetric Toeplitz matrix of shape [..., L, L]

    """
    vec_exp = torch.cat([torch.flip(vector, dims=(-1,)), vector[..., 1:]], dim=-1)
    v_len = vector.shape[-1]
    return torch.as_strided(vec_exp, size=vec_exp.shape[:-1] + (v_len, v_len), stride=vec_exp.stride()[:-1] + (1, 1)).flip(dims=(-1,))