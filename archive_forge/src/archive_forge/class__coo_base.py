from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast, upcast_char, to_native, isshape, getdtype,
import operator
class _coo_base(_data_matrix, _minmax_mixin):
    _format = 'coo'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)
        if isinstance(arg1, tuple):
            if isshape(arg1):
                M, N = arg1
                self._shape = check_shape((M, N))
                idx_dtype = self._get_index_dtype(maxval=max(M, N))
                data_dtype = getdtype(dtype, default=float)
                self.row = np.array([], dtype=idx_dtype)
                self.col = np.array([], dtype=idx_dtype)
                self.data = np.array([], dtype=data_dtype)
                self.has_canonical_format = True
            else:
                try:
                    obj, (row, col) = arg1
                except (TypeError, ValueError) as e:
                    raise TypeError('invalid input format') from e
                if shape is None:
                    if len(row) == 0 or len(col) == 0:
                        raise ValueError('cannot infer dimensions from zero sized index arrays')
                    M = operator.index(np.max(row)) + 1
                    N = operator.index(np.max(col)) + 1
                    self._shape = check_shape((M, N))
                else:
                    M, N = shape
                    self._shape = check_shape((M, N))
                idx_dtype = self._get_index_dtype((row, col), maxval=max(self.shape), check_contents=True)
                self.row = np.array(row, copy=copy, dtype=idx_dtype)
                self.col = np.array(col, copy=copy, dtype=idx_dtype)
                self.data = getdata(obj, copy=copy, dtype=dtype)
                self.has_canonical_format = False
        elif issparse(arg1):
            if arg1.format == self.format and copy:
                self.row = arg1.row.copy()
                self.col = arg1.col.copy()
                self.data = arg1.data.copy()
                self._shape = check_shape(arg1.shape)
            else:
                coo = arg1.tocoo()
                self.row = coo.row
                self.col = coo.col
                self.data = coo.data
                self._shape = check_shape(coo.shape)
            self.has_canonical_format = False
        else:
            M = np.atleast_2d(np.asarray(arg1))
            if M.ndim != 2:
                raise TypeError('expected dimension <= 2 array or matrix')
            self._shape = check_shape(M.shape)
            if shape is not None:
                if check_shape(shape) != self._shape:
                    message = f'inconsistent shapes: {shape} != {self._shape}'
                    raise ValueError(message)
            index_dtype = self._get_index_dtype(maxval=max(self._shape))
            row, col = M.nonzero()
            self.row = row.astype(index_dtype, copy=False)
            self.col = col.astype(index_dtype, copy=False)
            self.data = M[self.row, self.col]
            self.has_canonical_format = True
        if dtype is not None:
            self.data = self.data.astype(dtype, copy=False)
        self._check()

    def reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self
        nrows, ncols = self.shape
        if order == 'C':
            maxval = ncols * max(0, nrows - 1) + max(0, ncols - 1)
            dtype = self._get_index_dtype(maxval=maxval)
            flat_indices = np.multiply(ncols, self.row, dtype=dtype) + self.col
            new_row, new_col = divmod(flat_indices, shape[1])
        elif order == 'F':
            maxval = nrows * max(0, ncols - 1) + max(0, nrows - 1)
            dtype = self._get_index_dtype(maxval=maxval)
            flat_indices = np.multiply(nrows, self.col, dtype=dtype) + self.row
            new_col, new_row = divmod(flat_indices, shape[0])
        else:
            raise ValueError("'order' must be 'C' or 'F'")
        if copy:
            new_data = self.data.copy()
        else:
            new_data = self.data
        return self.__class__((new_data, (new_row, new_col)), shape=shape, copy=False)
    reshape.__doc__ = _spbase.reshape.__doc__

    def _getnnz(self, axis=None):
        if axis is None:
            nnz = len(self.data)
            if nnz != len(self.row) or nnz != len(self.col):
                raise ValueError('row, column, and data array must all be the same length')
            if self.data.ndim != 1 or self.row.ndim != 1 or self.col.ndim != 1:
                raise ValueError('row, column, and data arrays must be 1-D')
            return int(nnz)
        if axis < 0:
            axis += 2
        if axis == 0:
            return np.bincount(downcast_intp_index(self.col), minlength=self.shape[1])
        elif axis == 1:
            return np.bincount(downcast_intp_index(self.row), minlength=self.shape[0])
        else:
            raise ValueError('axis out of bounds')
    _getnnz.__doc__ = _spbase._getnnz.__doc__

    def _check(self):
        """ Checks data structure for consistency """
        if self.row.dtype.kind != 'i':
            warn(f'row index array has non-integer dtype ({self.row.dtype.name})', stacklevel=3)
        if self.col.dtype.kind != 'i':
            warn(f'col index array has non-integer dtype ({self.col.dtype.name})', stacklevel=3)
        idx_dtype = self._get_index_dtype((self.row, self.col), maxval=max(self.shape))
        self.row = np.asarray(self.row, dtype=idx_dtype)
        self.col = np.asarray(self.col, dtype=idx_dtype)
        self.data = to_native(self.data)
        if self.nnz > 0:
            if self.row.max() >= self.shape[0]:
                raise ValueError('row index exceeds matrix dimensions')
            if self.col.max() >= self.shape[1]:
                raise ValueError('column index exceeds matrix dimensions')
            if self.row.min() < 0:
                raise ValueError('negative row index found')
            if self.col.min() < 0:
                raise ValueError('negative column index found')

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse array/matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
        M, N = self.shape
        return self.__class__((self.data, (self.col, self.row)), shape=(N, M), copy=copy)
    transpose.__doc__ = _spbase.transpose.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        new_M, new_N = shape
        M, N = self.shape
        if new_M < M or new_N < N:
            mask = np.logical_and(self.row < new_M, self.col < new_N)
            if not mask.all():
                self.row = self.row[mask]
                self.col = self.col[mask]
                self.data = self.data[mask]
        self._shape = shape
    resize.__doc__ = _spbase.resize.__doc__

    def toarray(self, order=None, out=None):
        B = self._process_toarray_args(order, out)
        fortran = int(B.flags.f_contiguous)
        if not fortran and (not B.flags.c_contiguous):
            raise ValueError('Output array must be C or F contiguous')
        M, N = self.shape
        coo_todense(M, N, self.nnz, self.row, self.col, self.data, B.ravel('A'), fortran)
        return B
    toarray.__doc__ = _spbase.toarray.__doc__

    def tocsc(self, copy=False):
        """Convert this array/matrix to Compressed Sparse Column format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_array
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_array((data, (row, col)), shape=(4, 4)).tocsc()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.nnz == 0:
            return self._csc_container(self.shape, dtype=self.dtype)
        else:
            M, N = self.shape
            idx_dtype = self._get_index_dtype((self.col, self.row), maxval=max(self.nnz, M))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)
            indptr = np.empty(N + 1, dtype=idx_dtype)
            indices = np.empty_like(row, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))
            coo_tocsr(N, M, self.nnz, col, row, self.data, indptr, indices, data)
            x = self._csc_container((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocsr(self, copy=False):
        """Convert this array/matrix to Compressed Sparse Row format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_array
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_array((data, (row, col)), shape=(4, 4)).tocsr()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.nnz == 0:
            return self._csr_container(self.shape, dtype=self.dtype)
        else:
            M, N = self.shape
            idx_dtype = self._get_index_dtype((self.row, self.col), maxval=max(self.nnz, N))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)
            indptr = np.empty(M + 1, dtype=idx_dtype)
            indices = np.empty_like(col, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))
            coo_tocsr(M, N, self.nnz, row, col, self.data, indptr, indices, data)
            x = self._csr_container((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocoo(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self
    tocoo.__doc__ = _spbase.tocoo.__doc__

    def todia(self, copy=False):
        self.sum_duplicates()
        ks = self.col - self.row
        diags, diag_idx = np.unique(ks, return_inverse=True)
        if len(diags) > 100:
            warn('Constructing a DIA matrix with %d diagonals is inefficient' % len(diags), SparseEfficiencyWarning, stacklevel=2)
        if self.data.size == 0:
            data = np.zeros((0, 0), dtype=self.dtype)
        else:
            data = np.zeros((len(diags), self.col.max() + 1), dtype=self.dtype)
            data[diag_idx, self.col] = self.data
        return self._dia_container((data, diags), shape=self.shape)
    todia.__doc__ = _spbase.todia.__doc__

    def todok(self, copy=False):
        self.sum_duplicates()
        dok = self._dok_container(self.shape, dtype=self.dtype)
        dok._update(zip(zip(self.row, self.col), self.data))
        return dok
    todok.__doc__ = _spbase.todok.__doc__

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        diag = np.zeros(min(rows + min(k, 0), cols - max(k, 0)), dtype=self.dtype)
        diag_mask = self.row + k == self.col
        if self.has_canonical_format:
            row = self.row[diag_mask]
            data = self.data[diag_mask]
        else:
            row, _, data = self._sum_duplicates(self.row[diag_mask], self.col[diag_mask], self.data[diag_mask])
        diag[row + min(k, 0)] = data
        return diag
    diagonal.__doc__ = _data_matrix.diagonal.__doc__

    def _setdiag(self, values, k):
        M, N = self.shape
        if values.ndim and (not len(values)):
            return
        idx_dtype = self.row.dtype
        full_keep = self.col - self.row != k
        if k < 0:
            max_index = min(M + k, N)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.col >= max_index)
            new_row = np.arange(-k, -k + max_index, dtype=idx_dtype)
            new_col = np.arange(max_index, dtype=idx_dtype)
        else:
            max_index = min(M, N - k)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.row >= max_index)
            new_row = np.arange(max_index, dtype=idx_dtype)
            new_col = np.arange(k, k + max_index, dtype=idx_dtype)
        if values.ndim:
            new_data = values[:max_index]
        else:
            new_data = np.empty(max_index, dtype=self.dtype)
            new_data[:] = values
        self.row = np.concatenate((self.row[keep], new_row))
        self.col = np.concatenate((self.col[keep], new_col))
        self.data = np.concatenate((self.data[keep], new_data))
        self.has_canonical_format = False

    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the index arrays
        (i.e. .row and .col) are copied.
        """
        if copy:
            return self.__class__((data, (self.row.copy(), self.col.copy())), shape=self.shape, dtype=data.dtype)
        else:
            return self.__class__((data, (self.row, self.col)), shape=self.shape, dtype=data.dtype)

    def sum_duplicates(self):
        """Eliminate duplicate entries by adding them together

        This is an *in place* operation
        """
        if self.has_canonical_format:
            return
        summed = self._sum_duplicates(self.row, self.col, self.data)
        self.row, self.col, self.data = summed
        self.has_canonical_format = True

    def _sum_duplicates(self, row, col, data):
        if len(data) == 0:
            return (row, col, data)
        order = np.lexsort((col, row))
        row = row[order]
        col = col[order]
        data = data[order]
        unique_mask = (row[1:] != row[:-1]) | (col[1:] != col[:-1])
        unique_mask = np.append(True, unique_mask)
        row = row[unique_mask]
        col = col[unique_mask]
        unique_inds, = np.nonzero(unique_mask)
        data = np.add.reduceat(data, unique_inds, dtype=self.dtype)
        return (row, col, data)

    def eliminate_zeros(self):
        """Remove zero entries from the array/matrix

        This is an *in place* operation
        """
        mask = self.data != 0
        self.data = self.data[mask]
        self.row = self.row[mask]
        self.col = self.col[mask]

    def _add_dense(self, other):
        if other.shape != self.shape:
            raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        result = np.array(other, dtype=dtype, copy=True)
        fortran = int(result.flags.f_contiguous)
        M, N = self.shape
        coo_todense(M, N, self.nnz, self.row, self.col, self.data, result.ravel('A'), fortran)
        return self._container(result, copy=False)

    def _mul_vector(self, other):
        result = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char, other.dtype.char))
        coo_matvec(self.nnz, self.row, self.col, self.data, other, result)
        return result

    def _mul_multivector(self, other):
        result = np.zeros((other.shape[1], self.shape[0]), dtype=upcast_char(self.dtype.char, other.dtype.char))
        for i, col in enumerate(other.T):
            coo_matvec(self.nnz, self.row, self.col, self.data, col, result[i])
        return result.T.view(type=type(other))