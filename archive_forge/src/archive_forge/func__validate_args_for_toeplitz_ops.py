from warnings import warn
from itertools import product
import numpy as np
from numpy import atleast_1d, atleast_2d
from .lapack import get_lapack_funcs, _compute_lwork
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from ._decomp import _asarray_validated
from . import _decomp, _decomp_svd
from ._solve_toeplitz import levinson
from ._cythonized_array_utils import find_det_from_lu
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy.linalg._flinalg_py import get_flinalg_funcs  # noqa: F401
def _validate_args_for_toeplitz_ops(c_or_cr, b, check_finite, keep_b_shape, enforce_square=True):
    """Validate arguments and format inputs for toeplitz functions

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
        actual shape of ``c``, it will be converted to a 1-D array. If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``. Whatever the actual shape
        of ``r``, it will be converted to a 1-D array.
    b : (M,) or (M, K) array_like
        Right-hand side in ``T x = b``.
    check_finite : bool
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.
    keep_b_shape : bool
        Whether to convert a (M,) dimensional b into a (M, 1) dimensional
        matrix.
    enforce_square : bool, optional
        If True (default), this verifies that the Toeplitz matrix is square.

    Returns
    -------
    r : array
        1d array corresponding to the first row of the Toeplitz matrix.
    c: array
        1d array corresponding to the first column of the Toeplitz matrix.
    b: array
        (M,), (M, 1) or (M, K) dimensional array, post validation,
        corresponding to ``b``.
    dtype: numpy datatype
        ``dtype`` stores the datatype of ``r``, ``c`` and ``b``. If any of
        ``r``, ``c`` or ``b`` are complex, ``dtype`` is ``np.complex128``,
        otherwise, it is ``np.float``.
    b_shape: tuple
        Shape of ``b`` after passing it through ``_asarray_validated``.

    """
    if isinstance(c_or_cr, tuple):
        c, r = c_or_cr
        c = _asarray_validated(c, check_finite=check_finite).ravel()
        r = _asarray_validated(r, check_finite=check_finite).ravel()
    else:
        c = _asarray_validated(c_or_cr, check_finite=check_finite).ravel()
        r = c.conjugate()
    if b is None:
        raise ValueError('`b` must be an array, not None.')
    b = _asarray_validated(b, check_finite=check_finite)
    b_shape = b.shape
    is_not_square = r.shape[0] != c.shape[0]
    if enforce_square and is_not_square or b.shape[0] != r.shape[0]:
        raise ValueError('Incompatible dimensions.')
    is_cmplx = np.iscomplexobj(r) or np.iscomplexobj(c) or np.iscomplexobj(b)
    dtype = np.complex128 if is_cmplx else np.float64
    r, c, b = (np.asarray(i, dtype=dtype) for i in (r, c, b))
    if b.ndim == 1 and (not keep_b_shape):
        b = b.reshape(-1, 1)
    elif b.ndim != 1:
        b = b.reshape(b.shape[0], -1)
    return (r, c, b, dtype, b_shape)