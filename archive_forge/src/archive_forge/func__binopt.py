from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._data import _data_matrix, _minmax_mixin
from ._compressed import _cs_matrix
from ._base import issparse, _formats, _spbase, sparray
from ._sputils import (isshape, getdtype, getdata, to_native, upcast,
from . import _sparsetools
from ._sparsetools import (bsr_matvec, bsr_matvecs, csr_matmat_maxnnz,
def _binopt(self, other, op, in_shape=None, out_shape=None):
    """Apply the binary operation fn to two sparse matrices."""
    other = self.__class__(other, blocksize=self.blocksize)
    fn = getattr(_sparsetools, self.format + op + self.format)
    R, C = self.blocksize
    max_bnnz = len(self.data) + len(other.data)
    idx_dtype = self._get_index_dtype((self.indptr, self.indices, other.indptr, other.indices), maxval=max_bnnz)
    indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
    indices = np.empty(max_bnnz, dtype=idx_dtype)
    bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
    if op in bool_ops:
        data = np.empty(R * C * max_bnnz, dtype=np.bool_)
    else:
        data = np.empty(R * C * max_bnnz, dtype=upcast(self.dtype, other.dtype))
    fn(self.shape[0] // R, self.shape[1] // C, R, C, self.indptr.astype(idx_dtype), self.indices.astype(idx_dtype), self.data, other.indptr.astype(idx_dtype), other.indices.astype(idx_dtype), np.ravel(other.data), indptr, indices, data)
    actual_bnnz = indptr[-1]
    indices = indices[:actual_bnnz]
    data = data[:R * C * actual_bnnz]
    if actual_bnnz < max_bnnz / 2:
        indices = indices.copy()
        data = data.copy()
    data = data.reshape(-1, R, C)
    return self.__class__((data, indices, indptr), shape=self.shape)