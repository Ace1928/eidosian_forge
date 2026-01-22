import string
import warnings
import numpy
import cupy
import cupyx
from cupy import _core
from cupy._core import _scalar
from cupy._creation import basic
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _sputils
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _index
def _major_slice(self, idx, copy=False):
    """Index along the major axis where idx is a slice object.
        """
    M, N = self._swap(*self.shape)
    start, stop, step = idx.indices(M)
    if start == 0 and stop == M and (step == 1):
        return self.copy() if copy else self
    M = len(range(start, stop, step))
    new_shape = self._swap(M, N)
    if step == 1:
        if M == 0 or self.nnz == 0:
            return self.__class__(new_shape, dtype=self.dtype)
        return self.__class__(_index._get_csr_submatrix_major_axis(self.data, self.indices, self.indptr, start, stop), shape=new_shape, copy=copy)
    rows = cupy.arange(start, stop, step, dtype=self.indptr.dtype)
    return self._major_index_fancy(rows)