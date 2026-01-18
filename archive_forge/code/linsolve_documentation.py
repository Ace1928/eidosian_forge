from warnings import warn
import numpy as np
from numpy import asarray
from scipy.sparse import (issparse,
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.linalg import LinAlgError
import copy
from . import _superlu

    Solve the equation ``A x = b`` for `x`, assuming A is a triangular matrix.

    Parameters
    ----------
    A : (M, M) sparse matrix
        A sparse square triangular matrix. Should be in CSR format.
    b : (M,) or (M, N) array_like
        Right-hand side matrix in ``A x = b``
    lower : bool, optional
        Whether `A` is a lower or upper triangular matrix.
        Default is lower triangular matrix.
    overwrite_A : bool, optional
        Allow changing `A`. The indices of `A` are going to be sorted and zero
        entries are going to be removed.
        Enabling gives a performance gain. Default is False.
    overwrite_b : bool, optional
        Allow overwriting data in `b`.
        Enabling gives a performance gain. Default is False.
        If `overwrite_b` is True, it should be ensured that
        `b` has an appropriate dtype to be able to store the result.
    unit_diagonal : bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and will not be
        referenced.

        .. versionadded:: 1.4.0

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system ``A x = b``. Shape of return matches shape
        of `b`.

    Raises
    ------
    LinAlgError
        If `A` is singular or not triangular.
    ValueError
        If shape of `A` or shape of `b` do not match the requirements.

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.linalg import spsolve_triangular
    >>> A = csr_matrix([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
    >>> B = np.array([[2, 0], [-1, 0], [2, 0]], dtype=float)
    >>> x = spsolve_triangular(A, B)
    >>> np.allclose(A.dot(x), B)
    True
    