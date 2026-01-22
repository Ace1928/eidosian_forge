import ctypes
import warnings
import operator
from array import array as native_array
import numpy as np
from ..base import NotSupportedForSparseNDArray
from ..base import _LIB, numeric_types
from ..base import c_array_buf, mx_real_t, integer_types
from ..base import NDArrayHandle, check_call
from ..context import Context, current_context
from . import _internal
from . import op
from ._internal import _set_ndarray_class
from .ndarray import NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR, _int64_enabled
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array
from .ndarray import _ufunc_helper
class CSRNDArray(BaseSparseNDArray):
    """A sparse representation of 2D NDArray in the Compressed Sparse Row format.

    A CSRNDArray represents an NDArray as three separate arrays: `data`,
    `indptr` and `indices`. It uses the CSR representation where the column indices for
    row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their corresponding values are stored
    in ``data[indptr[i]:indptr[i+1]]``.

    The column indices for a given row are expected to be sorted in ascending order.
    Duplicate column entries for the same row are not allowed.

    Example
    -------
    >>> a = mx.nd.array([[0, 1, 0], [2, 0, 0], [0, 0, 0], [0, 0, 3]])
    >>> a = a.tostype('csr')
    >>> a.data.asnumpy()
    array([ 1.,  2.,  3.], dtype=float32)
    >>> a.indices.asnumpy()
    array([1, 0, 2])
    >>> a.indptr.asnumpy()
    array([0, 1, 2, 2, 3])

    See Also
    --------
    csr_matrix: Several ways to construct a CSRNDArray
    """

    def __reduce__(self):
        return (CSRNDArray, (None,), super(CSRNDArray, self).__getstate__())

    def __iadd__(self, other):
        (self + other).copyto(self)
        return self

    def __isub__(self, other):
        (self - other).copyto(self)
        return self

    def __imul__(self, other):
        (self * other).copyto(self)
        return self

    def __idiv__(self, other):
        (self / other).copyto(self)
        return self

    def __itruediv__(self, other):
        (self / other).copyto(self)
        return self

    def __getitem__(self, key):
        """x.__getitem__(i) <=> x[i]

        Returns a newly created NDArray based on the indexing key.

        Parameters
        ----------
        key : int or mxnet.ndarray.NDArray.slice
            Indexing key.

        Examples
        --------
        >>> indptr = np.array([0, 2, 3, 6])
        >>> indices = np.array([0, 2, 2, 0, 1, 2])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> a = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        >>> a.asnumpy()
        array([[ 1.,  0.,  2.],
               [ 0.,  0.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> a[1:2].asnumpy()
        array([[ 0.,  0.,  3.]], dtype=float32)
        >>> a[1].asnumpy()
        array([[ 0.,  0.,  3.]], dtype=float32)
        >>> a[-1].asnumpy()
        array([[ 4.,  5.,  6.]], dtype=float32)
        """
        if isinstance(key, int):
            if key == -1:
                begin = self.shape[0] - 1
            else:
                begin = key
            return op.slice(self, begin=begin, end=begin + 1)
        if isinstance(key, py_slice):
            if key.step is not None:
                raise ValueError('CSRNDArray only supports continuous slicing on axis 0')
            if key.start is not None or key.stop is not None:
                begin = key.start if key.start else 0
                end = key.stop if key.stop else self.shape[0]
                return op.slice(self, begin=begin, end=end)
            else:
                return self
        if isinstance(key, tuple):
            raise ValueError('Multi-dimension indexing is not supported')
        raise ValueError('Undefined behaviour for {}'.format(key))

    def __setitem__(self, key, value):
        """x.__setitem__(i, y) <=> x[i]=y

        Set self[key] to value. Only slice key [:] is supported.

        Parameters
        ----------
        key : mxnet.ndarray.NDArray.slice
            The indexing key.
        value : NDArray or CSRNDArray or numpy.ndarray
            The value to set.

        Examples
        --------
        >>> src = mx.nd.sparse.zeros('csr', (3,3))
        >>> src.asnumpy()
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
        >>> # assign CSRNDArray with same storage type
        >>> x = mx.nd.ones((3,3)).tostype('csr')
        >>> x[:] = src
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        >>> # assign NDArray to CSRNDArray
        >>> x[:] = mx.nd.ones((3,3)) * 2
        >>> x.asnumpy()
        array([[ 2.,  2.,  2.],
               [ 2.,  2.,  2.],
               [ 2.,  2.,  2.]], dtype=float32)
        """
        if not self.writable:
            raise ValueError('Failed to assign to a readonly CSRNDArray')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise ValueError('Assignment with slice for CSRNDArray is not implemented yet.')
            if isinstance(value, NDArray):
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                raise ValueError('Assigning numeric types to CSRNDArray is not implemented yet.')
            elif isinstance(value, (np.ndarray, np.generic)):
                warnings.warn('Assigning non-NDArray object to CSRNDArray is not efficient', RuntimeWarning)
                tmp = _array(value)
                tmp.copyto(self)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        else:
            assert isinstance(key, (int, tuple))
            raise Exception('CSRNDArray only supports [:] for assignment')

    @property
    def indices(self):
        """A deep copy NDArray of the indices array of the CSRNDArray.
        This generates a deep copy of the column indices of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's indices array.
        """
        return self._aux_data(1)

    @property
    def indptr(self):
        """A deep copy NDArray of the indptr array of the CSRNDArray.
        This generates a deep copy of the `indptr` of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's indptr array.
        """
        return self._aux_data(0)

    @property
    def data(self):
        """A deep copy NDArray of the data array of the CSRNDArray.
        This generates a deep copy of the `data` of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's data array.
        """
        return self._data()

    @indices.setter
    def indices(self, indices):
        raise NotImplementedError()

    @indptr.setter
    def indptr(self, indptr):
        raise NotImplementedError()

    @data.setter
    def data(self, data):
        raise NotImplementedError()

    def tostype(self, stype):
        """Return a copy of the array with chosen storage type.

        Returns
        -------
        NDArray or CSRNDArray
            A copy of the array with the chosen storage stype
        """
        if stype == 'row_sparse':
            raise ValueError('cast_storage from csr to row_sparse is not supported')
        return op.cast_storage(self, stype=stype)

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``NDArray`` or ``CSRNDArray`` object, then ``other.shape`` and
        ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``CSRNDArray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : NDArray or CSRNDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray or CSRNDArray
            The copied array. If ``other`` is an ``NDArray`` or ``CSRNDArray``, then the return
            value and ``other`` will point to the same ``NDArray`` or ``CSRNDArray``.
        """
        if isinstance(other, Context):
            return super(CSRNDArray, self).copyto(other)
        elif isinstance(other, NDArray):
            stype = other.stype
            if stype in ('default', 'csr'):
                return super(CSRNDArray, self).copyto(other)
            else:
                raise TypeError('copyto does not support destination NDArray stype ' + str(stype))
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def asscipy(self):
        """Returns a ``scipy.sparse.csr.csr_matrix`` object with value copied from this array

        Examples
        --------
        >>> x = mx.nd.sparse.zeros('csr', (2,3))
        >>> y = x.asscipy()
        >>> type(y)
        <type 'scipy.sparse.csr.csr_matrix'>
        >>> y
        <2x3 sparse matrix of type '<type 'numpy.float32'>'
        with 0 stored elements in Compressed Sparse Row format>
        """
        data = self.data.asnumpy()
        indices = self.indices.asnumpy()
        indptr = self.indptr.asnumpy()
        if not spsp:
            raise ImportError('scipy could not be imported. Please make sure that the scipy is installed.')
        return spsp.csr_matrix((data, indices, indptr), shape=self.shape, dtype=self.dtype)