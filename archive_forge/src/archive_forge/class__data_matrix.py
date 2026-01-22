import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _sputils
class _data_matrix(_base.spmatrix):

    def __init__(self, data):
        self.data = data

    @property
    def dtype(self):
        """Data type of the matrix."""
        return self.data.dtype

    def _with_data(self, data, copy=True):
        raise NotImplementedError

    def __abs__(self):
        """Elementwise abosulte."""
        return self._with_data(abs(self.data))

    def __neg__(self):
        """Elementwise negative."""
        return self._with_data(-self.data)

    def astype(self, t):
        """Casts the array to given data type.

        Args:
            dtype: Type specifier.

        Returns:
            A copy of the array with a given type.

        """
        return self._with_data(self.data.astype(t))

    def conj(self, copy=True):
        if cupy.issubdtype(self.dtype, cupy.complexfloating):
            return self._with_data(self.data.conj(), copy=copy)
        elif copy:
            return self.copy()
        else:
            return self
    conj.__doc__ = _base.spmatrix.conj.__doc__

    def copy(self):
        return self._with_data(self.data.copy(), copy=True)
    copy.__doc__ = _base.spmatrix.copy.__doc__

    def count_nonzero(self):
        """Returns number of non-zero entries.

        .. note::
           This method counts the actual number of non-zero entories, which
           does not include explicit zero entries.
           Instead ``nnz`` returns the number of entries including explicit
           zeros.

        Returns:
            Number of non-zero entries.

        """
        return cupy.count_nonzero(self.data)

    def mean(self, axis=None, dtype=None, out=None):
        """Compute the arithmetic mean along the specified axis.

        Args:
            axis (int or ``None``): Axis along which the sum is computed.
                If it is ``None``, it computes the average of all the elements.
                Select from ``{None, 0, 1, -2, -1}``.

        Returns:
            cupy.ndarray: Summed array.

        .. seealso::
           :meth:`scipy.sparse.spmatrix.mean`

        """
        _sputils.validateaxis(axis)
        nRow, nCol = self.shape
        data = self.data.copy()
        if axis is None:
            n = nRow * nCol
        elif axis in (0, -2):
            n = nRow
        else:
            n = nCol
        return self._with_data(data / n).sum(axis, dtype, out)

    def power(self, n, dtype=None):
        """Elementwise power function.

        Args:
            n: Exponent.
            dtype: Type specifier.

        """
        if dtype is None:
            data = self.data.copy()
        else:
            data = self.data.astype(dtype, copy=True)
        data **= n
        return self._with_data(data)