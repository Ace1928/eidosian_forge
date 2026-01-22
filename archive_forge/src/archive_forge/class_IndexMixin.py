import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
class IndexMixin(object):
    """
    This class provides common dispatching and validation logic for indexing.
    """

    def __getitem__(self, key):
        if scipy_available and numpy.lib.NumpyVersion(scipy.__version__) < '1.4.0':
            raise NotImplementedError('Sparse __getitem__() requires Scipy >= 1.4.0')
        row, col = self._parse_indices(key)
        if isinstance(row, _int_scalar_types):
            if isinstance(col, _int_scalar_types):
                return self._get_intXint(row, col)
            elif isinstance(col, slice):
                return self._get_intXslice(row, col)
            elif col.ndim == 1:
                return self._get_intXarray(row, col)
            raise IndexError('index results in >2 dimensions')
        elif isinstance(row, slice):
            if isinstance(col, _int_scalar_types):
                return self._get_sliceXint(row, col)
            elif isinstance(col, slice):
                if row == slice(None) and row == col:
                    return self.copy()
                return self._get_sliceXslice(row, col)
            elif col.ndim == 1:
                return self._get_sliceXarray(row, col)
            raise IndexError('index results in >2 dimensions')
        elif row.ndim == 1:
            if isinstance(col, _int_scalar_types):
                return self._get_arrayXint(row, col)
            elif isinstance(col, slice):
                return self._get_arrayXslice(row, col)
        elif isinstance(col, _int_scalar_types):
            return self._get_arrayXint(row, col)
        elif isinstance(col, slice):
            raise IndexError('index results in >2 dimensions')
        elif row.shape[1] == 1 and (col.ndim == 1 or col.shape[0] == 1):
            return self._get_columnXarray(row[:, 0], col.ravel())
        row, col = cupy.broadcast_arrays(row, col)
        if row.shape != col.shape:
            raise IndexError('number of row and column indices differ')
        if row.size == 0:
            return self.__class__(cupy.atleast_2d(row).shape, dtype=self.dtype)
        return self._get_arrayXarray(row, col)

    def __setitem__(self, key, x):
        row, col = self._parse_indices(key)
        if isinstance(row, _int_scalar_types) and isinstance(col, _int_scalar_types):
            x = cupy.asarray(x, dtype=self.dtype)
            if x.size != 1:
                raise ValueError('Trying to assign a sequence to an item')
            self._set_intXint(row, col, x.flat[0])
            return
        if isinstance(row, slice):
            row = cupy.arange(*row.indices(self.shape[0]))[:, None]
        else:
            row = cupy.atleast_1d(row)
        if isinstance(col, slice):
            col = cupy.arange(*col.indices(self.shape[1]))[None, :]
            if row.ndim == 1:
                row = row[:, None]
        else:
            col = cupy.atleast_1d(col)
        i, j = cupy.broadcast_arrays(row, col)
        if i.shape != j.shape:
            raise IndexError('number of row and column indices differ')
        if isspmatrix(x):
            if i.ndim == 1:
                i = i[None]
                j = j[None]
            broadcast_row = x.shape[0] == 1 and i.shape[0] != 1
            broadcast_col = x.shape[1] == 1 and i.shape[1] != 1
            if not ((broadcast_row or x.shape[0] == i.shape[0]) and (broadcast_col or x.shape[1] == i.shape[1])):
                raise ValueError('shape mismatch in assignment')
            if x.size == 0:
                return
            x = x.tocoo(copy=True)
            x.sum_duplicates()
            self._set_arrayXarray_sparse(i, j, x)
        else:
            x = cupy.asarray(x, dtype=self.dtype)
            x, _ = cupy.broadcast_arrays(x, i)
            if x.size == 0:
                return
            x = x.reshape(i.shape)
            self._set_arrayXarray(i, j, x)

    def _is_scalar(self, index):
        if isinstance(index, (cupy.ndarray, numpy.ndarray)) and index.ndim == 0 and (index.size == 1):
            return True
        return False

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

    def _asindices(self, idx, length):
        """Convert `idx` to a valid index for an axis with a given length.
        Subclasses that need special validation can override this method.

        idx is assumed to be at least a 1-dimensional array-like, but can
        have no more than 2 dimensions.
        """
        try:
            x = cupy.asarray(idx, dtype=self.indices.dtype)
        except (ValueError, TypeError, MemoryError):
            raise IndexError('invalid index')
        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be <= 2')
        return x % length

    def getrow(self, i):
        """Return a copy of row i of the matrix, as a (1 x n) row vector.

        Args:
            i (integer): Row

        Returns:
            cupyx.scipy.sparse.spmatrix: Sparse matrix with single row
        """
        M, N = self.shape
        i = _normalize_index(i, M, 'index')
        return self._get_intXslice(i, slice(None))

    def getcol(self, i):
        """Return a copy of column i of the matrix, as a (m x 1) column vector.

        Args:
            i (integer): Column

        Returns:
            cupyx.scipy.sparse.spmatrix: Sparse matrix with single column
        """
        M, N = self.shape
        i = _normalize_index(i, N, 'index')
        return self._get_sliceXint(slice(None), i)

    def _get_intXint(self, row, col):
        raise NotImplementedError()

    def _get_intXarray(self, row, col):
        raise NotImplementedError()

    def _get_intXslice(self, row, col):
        raise NotImplementedError()

    def _get_sliceXint(self, row, col):
        raise NotImplementedError()

    def _get_sliceXslice(self, row, col):
        raise NotImplementedError()

    def _get_sliceXarray(self, row, col):
        raise NotImplementedError()

    def _get_arrayXint(self, row, col):
        raise NotImplementedError()

    def _get_arrayXslice(self, row, col):
        raise NotImplementedError()

    def _get_columnXarray(self, row, col):
        raise NotImplementedError()

    def _get_arrayXarray(self, row, col):
        raise NotImplementedError()

    def _set_intXint(self, row, col, x):
        raise NotImplementedError()

    def _set_arrayXarray(self, row, col, x):
        raise NotImplementedError()

    def _set_arrayXarray_sparse(self, row, col, x):
        x = cupy.asarray(x.toarray(), dtype=self.dtype)
        x, _ = cupy.broadcast_arrays(x, row)
        self._set_arrayXarray(row, col, x)