import torch
from .utils import _csr_to_coo, _transpose_with_info
def _sddmm_func(a, b, row_indices, row_offsets, column_indices):
    sparsity = 1 - column_indices.shape[0] / (a.shape[1] * b.shape[1])
    if _should_use_coo(a, sparsity):
        m = a.shape[-2]
        n = b.shape[-2]
        ro, ci = _csr_to_coo(m, n, row_offsets, column_indices)
        return torch.ops.xformers.coo_sddmm(a, b, row_indices, ro, ci)
    elif _should_use_csr_ge(a, sparsity):
        return torch.ops.xformers.csr_sddmm(a, b, row_indices, row_offsets, column_indices)
    return torch.ops.xformers.sddmm_sputnik(a, b, row_indices, row_offsets, column_indices)