from bisect import bisect_left
import numpy as np
from ._matrix import spmatrix
from ._base import _spbase, sparray, issparse
from ._index import IndexMixin, INT_TYPES, _broadcast_arrays
from ._sputils import (getdtype, isshape, isscalarlike, upcast_scalar,
from . import _csparsetools
def _get_row_ranges(self, rows, col_slice):
    """
        Fast path for indexing in the case where column index is slice.

        This gains performance improvement over brute force by more
        efficient skipping of zeros, by accessing the elements
        column-wise in order.

        Parameters
        ----------
        rows : sequence or range
            Rows indexed. If range, must be within valid bounds.
        col_slice : slice
            Columns indexed

        """
    j_start, j_stop, j_stride = col_slice.indices(self.shape[1])
    col_range = range(j_start, j_stop, j_stride)
    nj = len(col_range)
    new = self._lil_container((len(rows), nj), dtype=self.dtype)
    _csparsetools.lil_get_row_ranges(self.shape[0], self.shape[1], self.rows, self.data, new.rows, new.data, rows, j_start, j_stop, j_stride, nj)
    return new