import numbers
import math
import numpy as np
from scipy._lib._util import check_random_state, rng_integers
from ._sputils import upcast, get_index_dtype, isscalarlike
from ._sparsetools import csr_hstack
from ._bsr import bsr_matrix, bsr_array
from ._coo import coo_matrix, coo_array
from ._csc import csc_matrix, csc_array
from ._csr import csr_matrix, csr_array
from ._dia import dia_matrix, dia_array
from ._base import issparse, sparray
def _stack_along_minor_axis(blocks, axis):
    """
    Stacking fast path for CSR/CSC matrices along the minor axis
    (i) hstack for CSR, (ii) vstack for CSC.
    """
    n_blocks = len(blocks)
    if n_blocks == 0:
        raise ValueError('Missing block matrices')
    if n_blocks == 1:
        return blocks[0]
    other_axis = 1 if axis == 0 else 0
    other_axis_dims = {b.shape[other_axis] for b in blocks}
    if len(other_axis_dims) > 1:
        raise ValueError(f'Mismatching dimensions along axis {other_axis}: {other_axis_dims}')
    constant_dim, = other_axis_dims
    indptr_list = [b.indptr for b in blocks]
    data_cat = np.concatenate([b.data for b in blocks])
    sum_dim = sum((b.shape[axis] for b in blocks))
    nnz = sum((len(b.indices) for b in blocks))
    idx_dtype = get_index_dtype(maxval=max(sum_dim - 1, nnz))
    stack_dim_cat = np.array([b.shape[axis] for b in blocks], dtype=idx_dtype)
    if data_cat.size > 0:
        indptr_cat = np.concatenate(indptr_list).astype(idx_dtype)
        indices_cat = np.concatenate([b.indices for b in blocks]).astype(idx_dtype)
        indptr = np.empty(constant_dim + 1, dtype=idx_dtype)
        indices = np.empty_like(indices_cat)
        data = np.empty_like(data_cat)
        csr_hstack(n_blocks, constant_dim, stack_dim_cat, indptr_cat, indices_cat, data_cat, indptr, indices, data)
    else:
        indptr = np.zeros(constant_dim + 1, dtype=idx_dtype)
        indices = np.empty(0, dtype=idx_dtype)
        data = np.empty(0, dtype=data_cat.dtype)
    if axis == 0:
        return blocks[0]._csc_container((data, indices, indptr), shape=(sum_dim, constant_dim))
    else:
        return blocks[0]._csr_container((data, indices, indptr), shape=(constant_dim, sum_dim))