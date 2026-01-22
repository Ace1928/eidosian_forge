from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._data import _data_matrix, _minmax_mixin
from ._compressed import _cs_matrix
from ._base import issparse, _formats, _spbase, sparray
from ._sputils import (isshape, getdtype, getdata, to_native, upcast,
from . import _sparsetools
from ._sparsetools import (bsr_matvec, bsr_matvecs, csr_matmat_maxnnz,
class bsr_array(_bsr_base, sparray):
    """
    Block Sparse Row format sparse array.

    This can be instantiated in several ways:
        bsr_array(D, [blocksize=(R,C)])
            where D is a 2-D ndarray.

        bsr_array(S, [blocksize=(R,C)])
            with another sparse array or matrix S (equivalent to S.tobsr())

        bsr_array((M, N), [blocksize=(R,C), dtype])
            to construct an empty sparse array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        bsr_array((data, ij), [blocksize=(R,C), shape=(M, N)])
            where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``

        bsr_array((data, indices, indptr), [shape=(M, N)])
            is the standard BSR representation where the block column
            indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding block values are stored in
            ``data[ indptr[i]: indptr[i+1] ]``. If the shape parameter is not
            supplied, the array dimensions are inferred from the index arrays.

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
        BSR format data array of the array
    indices
        BSR format index array of the array
    indptr
        BSR format index pointer array of the array
    blocksize
        Block size
    has_sorted_indices : bool
        Whether indices are sorted
    has_canonical_format : bool
    T

    Notes
    -----
    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    **Summary of BSR format**

    The Block Sparse Row (BSR) format is very similar to the Compressed
    Sparse Row (CSR) format. BSR is appropriate for sparse matrices with dense
    sub matrices like the last example below. Such sparse block matrices often
    arise in vector-valued finite element discretizations. In such cases, BSR is
    considerably more efficient than CSR and CSC for many sparse arithmetic
    operations.

    **Blocksize**

    The blocksize (R,C) must evenly divide the shape of the sparse array (M,N).
    That is, R and C must satisfy the relationship ``M % R = 0`` and
    ``N % C = 0``.

    If no blocksize is specified, a simple heuristic is applied to determine
    an appropriate blocksize.

    **Canonical Format**

    In canonical format, there are no duplicate blocks and indices are sorted
    per row.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import bsr_array
    >>> bsr_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3 ,4, 5, 6])
    >>> bsr_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    >>> bsr_array((data,indices,indptr), shape=(6, 6)).toarray()
    array([[1, 1, 0, 0, 2, 2],
           [1, 1, 0, 0, 2, 2],
           [0, 0, 0, 0, 3, 3],
           [0, 0, 0, 0, 3, 3],
           [4, 4, 5, 5, 6, 6],
           [4, 4, 5, 5, 6, 6]])

    """