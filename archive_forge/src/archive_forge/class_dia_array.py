import numpy as np
from ._matrix import spmatrix
from ._base import issparse, _formats, _spbase, sparray
from ._data import _data_matrix
from ._sputils import (
from ._sparsetools import dia_matvec
class dia_array(_dia_base, sparray):
    """
    Sparse array with DIAgonal storage.

    This can be instantiated in several ways:
        dia_array(D)
            where D is a 2-D ndarray

        dia_array(S)
            with another sparse array or matrix S (equivalent to S.todia())

        dia_array((M, N), [dtype])
            to construct an empty array with shape (M, N),
            dtype is optional, defaulting to dtype='d'.

        dia_array((data, offsets), shape=(M, N))
            where the ``data[k,:]`` stores the diagonal entries for
            diagonal ``offsets[k]`` (See example below)

    Attributes
    ----------
    dtype : dtype
        Data type of the array
    shape : 2-tuple
        Shape of the array
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        DIA format data array of the array
    offsets
        DIA format offset array of the array
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import dia_array
    >>> dia_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> offsets = np.array([0, -1, 2])
    >>> dia_array((data, offsets), shape=(4, 4)).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    >>> from scipy.sparse import dia_array
    >>> n = 10
    >>> ex = np.ones(n)
    >>> data = np.array([ex, 2 * ex, ex])
    >>> offsets = np.array([-1, 0, 1])
    >>> dia_array((data, offsets), shape=(n, n)).toarray()
    array([[2., 1., 0., ..., 0., 0., 0.],
           [1., 2., 1., ..., 0., 0., 0.],
           [0., 1., 2., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 2., 1., 0.],
           [0., 0., 0., ..., 1., 2., 1.],
           [0., 0., 0., ..., 0., 1., 2.]])
    """