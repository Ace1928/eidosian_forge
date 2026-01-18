import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def tf2ss(num, den):
    """Transfer function to state-space representation.

    Parameters
    ----------
    num, den : array_like
        Sequences representing the coefficients of the numerator and
        denominator polynomials, in order of descending degree. The
        denominator needs to be at least as long as the numerator.

    Returns
    -------
    A, B, C, D : ndarray
        State space representation of the system, in controller canonical
        form.

    See Also
    --------
    scipy.signal.tf2ss
    """
    num, den = normalize(num, den)
    nn = len(num.shape)
    if nn == 1:
        num = cupy.asarray([num], num.dtype)
    M = num.shape[1]
    K = len(den)
    if M > K:
        msg = 'Improper transfer function. `num` is longer than `den`.'
        raise ValueError(msg)
    if M == 0 or K == 0:
        return (cupy.array([], float), cupy.array([], float), cupy.array([], float), cupy.array([], float))
    num = cupy.hstack((cupy.zeros((num.shape[0], K - M), num.dtype), num))
    if num.shape[-1] > 0:
        D = cupy.atleast_2d(num[:, 0])
    else:
        D = cupy.array([[0]], float)
    if K == 1:
        D = D.reshape(num.shape)
        return (cupy.zeros((1, 1)), cupy.zeros((1, D.shape[1])), cupy.zeros((D.shape[0], 1)), D)
    frow = -cupy.array([den[1:]])
    A = cupy.r_[frow, cupy.eye(K - 2, K - 1)]
    B = cupy.eye(K - 1, 1)
    C = num[:, 1:] - cupy.outer(num[:, 0], den[1:])
    D = D.reshape((C.shape[0], B.shape[1]))
    return (A, B, C, D)