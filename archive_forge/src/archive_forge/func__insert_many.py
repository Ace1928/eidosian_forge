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
def _insert_many(self, i, j, x):
    """Inserts new nonzero at each (i, j) with value x
        Here (i,j) index major and minor respectively.
        i, j and x must be non-empty, 1d arrays.
        Inserts each major group (e.g. all entries per row) at a time.
        Maintains has_sorted_indices property.
        Modifies i, j, x in place.
        """
    order = cupy.argsort(i)
    i = i.take(order)
    j = j.take(order)
    x = x.take(order)
    idx_dtype = _sputils.get_index_dtype((self.indices, self.indptr), maxval=self.nnz + x.size)
    self.indptr = self.indptr.astype(idx_dtype)
    self.indices = self.indices.astype(idx_dtype)
    self.data = self.data.astype(self.dtype)
    indptr_inserts, indices_inserts, data_inserts = _index._select_last_indices(i, j, x, idx_dtype)
    rows, ui_indptr = cupy.unique(indptr_inserts, return_index=True)
    to_add = cupy.empty(ui_indptr.size + 1, ui_indptr.dtype)
    to_add[-1] = j.size
    to_add[:-1] = ui_indptr
    ui_indptr = to_add
    row_counts = cupy.zeros(ui_indptr.size - 1, dtype=idx_dtype)
    cupy.add.at(row_counts, cupy.searchsorted(rows, indptr_inserts), 1)
    self._perform_insert(indices_inserts, data_inserts, rows, row_counts, idx_dtype)