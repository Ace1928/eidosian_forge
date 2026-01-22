import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _parse_indices(self, key):
    M, N = self.shape
    row, col = _unpack_index(key)
    if self._is_scalar(row):
        row = row.item()
    if self._is_scalar(col):
        col = col.item()
    if isinstance(row, _int_scalar_types):
        row = _normalize_index(row, M, 'row')
    elif not isinstance(row, slice):
        row = self._asindices(row, M)
    if isinstance(col, _int_scalar_types):
        col = _normalize_index(col, N, 'column')
    elif not isinstance(col, slice):
        col = self._asindices(col, N)
    return (row, col)