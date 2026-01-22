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
def __get_sorted(self):
    """Determine whether the matrix has sorted indices.

        Returns
            bool:
                ``True`` if the indices of the matrix are in sorted order,
                otherwise ``False``.

        .. warning::
            Getting this property might synchronize the device.

        """
    if self.data.size == 0:
        self._has_sorted_indices = True
    elif not hasattr(self, '_has_sorted_indices'):
        is_sorted = self._has_sorted_indices_kern(self.indptr, self.indices, size=self.indptr.size - 1)
        self._has_sorted_indices = bool(is_sorted.all())
    return self._has_sorted_indices