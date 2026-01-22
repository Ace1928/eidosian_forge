import torch
def _coo_to_csr(m, n, row_indices, column_indices):
    row_offsets = row_indices.bincount(minlength=n).cumsum(0, dtype=row_indices.dtype)
    row_offsets = torch.nn.functional.pad(row_offsets, (1, 0))
    return (row_offsets, column_indices)