import torch
def _csr_to_coo(m, n, row_offsets, column_indices):
    indices = torch.arange(m, dtype=row_offsets.dtype, device=row_offsets.device)
    row_sizes = torch.diff(row_offsets)
    row_coo = torch.repeat_interleave(indices, row_sizes.long())
    return (row_coo, column_indices)