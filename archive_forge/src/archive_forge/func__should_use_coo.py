import torch
from .utils import _csr_to_coo, _transpose_with_info
def _should_use_coo(a, sparsity):
    if not a.is_cuda:
        return False
    B, M, K = a.shape
    if B < 32 and M < 4096:
        return False
    if sparsity > 0.995:
        return False
    if sparsity < 0.9:
        return False
    if K > 64:
        return False
    return sparsity > 0.97