import numpy as np
from warnings import warn
from ._sputils import isintlike
def _validate_indices(self, key):
    M, N = self.shape
    row, col = _unpack_index(key)
    if isintlike(row):
        row = int(row)
        if row < -M or row >= M:
            raise IndexError('row index (%d) out of range' % row)
        if row < 0:
            row += M
    elif not isinstance(row, slice):
        row = self._asindices(row, M)
    if isintlike(col):
        col = int(col)
        if col < -N or col >= N:
            raise IndexError('column index (%d) out of range' % col)
        if col < 0:
            col += N
    elif not isinstance(col, slice):
        col = self._asindices(col, N)
    return (row, col)