import torch
def _get_transpose_info(m, n, row_indices, row_offsets, column_indices):
    row_coo, _ = _csr_to_coo(m, n, row_offsets, column_indices)
    row_offsets_t, perm = column_indices.sort(dim=0, stable=True)
    column_indices_t = row_coo[perm]
    row_offsets_t, _ = _coo_to_csr(m, n, row_offsets_t, column_indices)
    row_indices_t = _diffsort(row_offsets_t).int()
    return (row_indices_t, row_offsets_t, column_indices_t, perm)