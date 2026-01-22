import functools
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator
from ..sparse import issparse, csc_matrix, csr_matrix, coo_matrix, find
from ._group_columns import group_dense, group_sparse
from scipy._lib._array_api import atleast_nd, array_namespace
@functools.lru_cache
def _eps_for_method(x0_dtype, f0_dtype, method):
    """
    Calculates relative EPS step to use for a given data type
    and numdiff step method.

    Progressively smaller steps are used for larger floating point types.

    Parameters
    ----------
    f0_dtype: np.dtype
        dtype of function evaluation

    x0_dtype: np.dtype
        dtype of parameter vector

    method: {'2-point', '3-point', 'cs'}

    Returns
    -------
    EPS: float
        relative step size. May be np.float16, np.float32, np.float64

    Notes
    -----
    The default relative step will be np.float64. However, if x0 or f0 are
    smaller floating point types (np.float16, np.float32), then the smallest
    floating point type is chosen.
    """
    EPS = np.finfo(np.float64).eps
    x0_is_fp = False
    if np.issubdtype(x0_dtype, np.inexact):
        EPS = np.finfo(x0_dtype).eps
        x0_itemsize = np.dtype(x0_dtype).itemsize
        x0_is_fp = True
    if np.issubdtype(f0_dtype, np.inexact):
        f0_itemsize = np.dtype(f0_dtype).itemsize
        if x0_is_fp and f0_itemsize < x0_itemsize:
            EPS = np.finfo(f0_dtype).eps
    if method in ['2-point', 'cs']:
        return EPS ** 0.5
    elif method in ['3-point']:
        return EPS ** (1 / 3)
    else:
        raise RuntimeError("Unknown step method, should be one of {'2-point', '3-point', 'cs'}")