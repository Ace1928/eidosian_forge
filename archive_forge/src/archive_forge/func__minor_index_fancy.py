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
def _minor_index_fancy(self, idx):
    """Index along the minor axis where idx is an array of ints.
        """
    M, _ = self._swap(*self.shape)
    N = idx.size
    new_shape = self._swap(M, N)
    if self.nnz == 0 or N == 0:
        return self.__class__(new_shape, dtype=self.dtype)
    if idx.size * M < self.nnz:
        pass
    return self._tocsx()._major_index_fancy(idx)._tocsx()