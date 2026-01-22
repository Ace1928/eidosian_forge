import torch
def _sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
    if sparse.dim() != 2:
        raise RuntimeError(f'Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor')
    m, k = sparse.shape
    device = sparse.device
    if meta_reordered.dim() != 2:
        raise RuntimeError(f'Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor')
    if meta_reordered.device != device:
        raise RuntimeError(f'Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device')
    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f'Invalid datatype {meta_dtype} of meta matrix')
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(f'Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}')
    if meta_ncols * 4 * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(f'Number of columns of sparse matrix {k} different from the {meta_ncols * 4 * quadbits_per_meta_elem // 2}, expected according to the number of columns of meta matrix')
    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = [0, 1, 32, 33]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = [0, 1, 4, 5]
    tmp1 = torch.tensor([0, 2], dtype=torch.int64, device=device).repeat(meta_nrows, meta_ncols // 2)
    tmp2 = (torch.arange(0, meta_ncols // 2, device=device) * 2 * meta_nrows).view(-1, 1).repeat(1, 2).view(-1).repeat(m, 1)
    tmp3 = (torch.arange(0, 8, device=device) * magic0).view(-1, 1).repeat(m // 8, meta_ncols)
    tmp4 = torch.tensor(magic1, device=device).view(-1, 1).repeat(1, 8 * meta_ncols).repeat(meta_nrows // 32, 1).view(meta_nrows, meta_ncols)
    tmp5 = (torch.arange(0, meta_nrows // 32, device=device) * 64).view(-1, 1).repeat(1, 32 * meta_ncols).view(meta_nrows, meta_ncols)
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets.view(-1)).view(m, meta_ncols)
    meta_2 = torch.empty((m, meta_ncols, 2 * quadbits_per_meta_elem), dtype=meta_dtype, device=device)
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 3
        meta_2[:, :, 1] = meta >> 2 & 3
        meta_2[:, :, 2] = meta >> 4 & 3
        meta_2[:, :, 3] = meta >> 6 & 3
        meta_2[:, :, 4] = meta >> 8 & 3
        meta_2[:, :, 5] = meta >> 10 & 3
        meta_2[:, :, 6] = meta >> 12 & 3
        meta_2[:, :, 7] = meta >> 14 & 3
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 3
        meta_2[:, :, 1] = meta >> 2 & 3
        meta_2[:, :, 2] = meta >> 4 & 3
        meta_2[:, :, 3] = meta >> 6 & 3
        meta_2[:, :, 4] = meta >> 8 & 3
        meta_2[:, :, 5] = meta >> 10 & 3
        meta_2[:, :, 6] = meta >> 12 & 3
        meta_2[:, :, 7] = meta >> 14 & 3
        meta_2[:, :, 8] = meta >> 16 & 3
        meta_2[:, :, 9] = meta >> 18 & 3
        meta_2[:, :, 10] = meta >> 20 & 3
        meta_2[:, :, 11] = meta >> 22 & 3
        meta_2[:, :, 12] = meta >> 24 & 3
        meta_2[:, :, 13] = meta >> 26 & 3
        meta_2[:, :, 14] = meta >> 28 & 3
        meta_2[:, :, 15] = meta >> 30 & 3
    dense_offsets = meta_2.view(-1) + (torch.arange(0, m * k // 2, device=device) * 4).view(-1, 1).repeat(1, 2).view(-1)
    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    dense.scatter_(0, dense_offsets, sparse.view(-1))
    return dense.view(m, 2 * k)