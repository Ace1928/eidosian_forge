import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
def _valid_inputs(A, B, poles, method, rtol, maxiter):
    """
    Check the poles come in complex conjugage pairs
    Check shapes of A, B and poles are compatible.
    Check the method chosen is compatible with provided poles
    Return update method to use and ordered poles

    """
    if poles.ndim > 1:
        raise ValueError('Poles must be a 1D array like.')
    poles = _order_complex_poles(poles)
    if A.ndim > 2:
        raise ValueError('A must be a 2D array/matrix.')
    if B.ndim > 2:
        raise ValueError('B must be a 2D array/matrix')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be square')
    if len(poles) > A.shape[0]:
        raise ValueError('maximum number of poles is %d but you asked for %d' % (A.shape[0], len(poles)))
    if len(poles) < A.shape[0]:
        raise ValueError('number of poles is %d but you should provide %d' % (len(poles), A.shape[0]))
    r = cupy.linalg.matrix_rank(B)
    for p in poles:
        if sum(p == poles) > r:
            raise ValueError('at least one of the requested pole is repeated more than rank(B) times')
    update_loop = _YT_loop
    if method not in ('KNV0', 'YT'):
        raise ValueError("The method keyword must be one of 'YT' or 'KNV0'")
    if method == 'KNV0':
        update_loop = _KNV0_loop
        if not all(cupy.isreal(poles)):
            raise ValueError('Complex poles are not supported by KNV0')
    if maxiter < 1:
        raise ValueError('maxiter must be at least equal to 1')
    if rtol > 1:
        raise ValueError('rtol can not be greater than 1')
    return (update_loop, poles)