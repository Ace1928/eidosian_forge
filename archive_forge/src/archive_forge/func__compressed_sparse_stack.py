import numpy
import cupy
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _dia
from cupyx.scipy.sparse import _sputils
def _compressed_sparse_stack(blocks, axis):
    """Fast path for stacking CSR/CSC matrices
    (i) vstack for CSR, (ii) hstack for CSC.
    """
    other_axis = 1 if axis == 0 else 0
    data = cupy.concatenate([b.data for b in blocks])
    constant_dim = blocks[0].shape[other_axis]
    idx_dtype = _sputils.get_index_dtype(arrays=[b.indptr for b in blocks], maxval=max(data.size, constant_dim))
    indices = cupy.empty(data.size, dtype=idx_dtype)
    indptr = cupy.empty(sum((b.shape[axis] for b in blocks)) + 1, dtype=idx_dtype)
    last_indptr = idx_dtype(0)
    sum_dim = 0
    sum_indices = 0
    for b in blocks:
        if b.shape[other_axis] != constant_dim:
            raise ValueError('incompatible dimensions for axis %d' % other_axis)
        indices[sum_indices:sum_indices + b.indices.size] = b.indices
        sum_indices += b.indices.size
        idxs = slice(sum_dim, sum_dim + b.shape[axis])
        indptr[idxs] = b.indptr[:-1]
        indptr[idxs] += last_indptr
        sum_dim += b.shape[axis]
        last_indptr += b.indptr[-1]
    indptr[-1] = last_indptr
    if axis == 0:
        return _csr.csr_matrix((data, indices, indptr), shape=(sum_dim, constant_dim))
    else:
        return _csc.csc_matrix((data, indices, indptr), shape=(constant_dim, sum_dim))