from functools import update_wrapper
from numbers import Number
from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch.overrides import is_tensor_like
def tril_matrix_to_vec(mat: torch.Tensor, diag: int=0) -> torch.Tensor:
    """
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
    n = mat.shape[-1]
    if not torch._C._get_tracing_state() and (diag < -n or diag >= n):
        raise ValueError(f'diag ({diag}) provided is outside [{-n}, {n - 1}].')
    arange = torch.arange(n, device=mat.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    vec = mat[..., tril_mask]
    return vec