import operator
import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
from . import _specfun
from ._comb import _comb_int
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def ai_zeros(nt):
    """
    Compute `nt` zeros and values of the Airy function Ai and its derivative.

    Computes the first `nt` zeros, `a`, of the Airy function Ai(x);
    first `nt` zeros, `ap`, of the derivative of the Airy function Ai'(x);
    the corresponding values Ai(a');
    and the corresponding values Ai'(a).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    a : ndarray
        First `nt` zeros of Ai(x)
    ap : ndarray
        First `nt` zeros of Ai'(x)
    ai : ndarray
        Values of Ai(x) evaluated at first `nt` zeros of Ai'(x)
    aip : ndarray
        Values of Ai'(x) evaluated at first `nt` zeros of Ai(x)

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> from scipy import special
    >>> a, ap, ai, aip = special.ai_zeros(3)
    >>> a
    array([-2.33810741, -4.08794944, -5.52055983])
    >>> ap
    array([-1.01879297, -3.24819758, -4.82009921])
    >>> ai
    array([ 0.53565666, -0.41901548,  0.38040647])
    >>> aip
    array([ 0.70121082, -0.80311137,  0.86520403])

    """
    kf = 1
    if not isscalar(nt) or floor(nt) != nt or nt <= 0:
        raise ValueError('nt must be a positive integer scalar.')
    return _specfun.airyzo(nt, kf)