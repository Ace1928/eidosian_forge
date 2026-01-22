import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def _read_body_coo(cursor, generalize_symmetry=True):
    """
    Read MatrixMarket coordinate body
    """
    from . import _fmm_core
    index_dtype = 'int32'
    if cursor.header.nrows >= 2 ** 31 or cursor.header.ncols >= 2 ** 31:
        index_dtype = 'int64'
    i = np.zeros(cursor.header.nnz, dtype=index_dtype)
    j = np.zeros(cursor.header.nnz, dtype=index_dtype)
    data = np.zeros(cursor.header.nnz, dtype=_field_to_dtype.get(cursor.header.field))
    _fmm_core.read_body_coo(cursor, i, j, data)
    if generalize_symmetry and cursor.header.symmetry != 'general':
        off_diagonal_mask = i != j
        off_diagonal_rows = i[off_diagonal_mask]
        off_diagonal_cols = j[off_diagonal_mask]
        off_diagonal_data = data[off_diagonal_mask]
        if cursor.header.symmetry == 'skew-symmetric':
            off_diagonal_data *= -1
        elif cursor.header.symmetry == 'hermitian':
            off_diagonal_data = off_diagonal_data.conjugate()
        i = np.concatenate((i, off_diagonal_cols))
        j = np.concatenate((j, off_diagonal_rows))
        data = np.concatenate((data, off_diagonal_data))
    return ((data, (i, j)), cursor.header.shape)