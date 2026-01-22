from warnings import warn
import numpy as np
from numpy import (atleast_2d, arange, zeros_like, imag, diag,
from scipy._lib._util import ComplexWarning
from ._decomp import _asarray_validated
from .lapack import get_lapack_funcs, _compute_lwork

    Helper function to construct explicit outer factors of LDL factorization.

    If lower is True the permuted factors are multiplied as L(1)*L(2)*...*L(k).
    Otherwise, the permuted factors are multiplied as L(k)*...*L(2)*L(1). See
    LAPACK documentation for more details.

    Parameters
    ----------
    lu : ndarray
        The triangular array that is extracted from LAPACK routine call with
        ones on the diagonals.
    swap_vec : ndarray
        The array that defines the row swapping indices. If the kth entry is m
        then rows k,m are swapped. Notice that the mth entry is not necessarily
        k to avoid undoing the swapping.
    pivs : ndarray
        The array that defines the block diagonal structure returned by
        _ldl_sanitize_ipiv().
    lower : bool, optional
        The boolean to switch between lower and upper triangular structure.

    Returns
    -------
    lu : ndarray
        The square outer factor which satisfies the L * D * L.T = A
    perm : ndarray
        The permutation vector that brings the lu to the triangular form

    Notes
    -----
    Note that the original argument "lu" is overwritten.

    