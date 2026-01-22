import math
import operator
import warnings
import numpy
import numpy as np
from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
from numpy.polynomial.polynomial import polyval as npp_polyval
from numpy.polynomial.polynomial import polyvalfromroots
from scipy import special, optimize, fft as sp_fft
from scipy.special import comb
from scipy._lib._util import float_factorial
def _campos_zeros(n):
    """
    Return approximate zero locations of Bessel polynomials y_n(x) for order
    `n` using polynomial fit (Campos-Calderon 2011)
    """
    if n == 1:
        return asarray([-1 + 0j])
    s = npp_polyval(n, [0, 0, 2, 0, -3, 1])
    b3 = npp_polyval(n, [16, -8]) / s
    b2 = npp_polyval(n, [-24, -12, 12]) / s
    b1 = npp_polyval(n, [8, 24, -12, -2]) / s
    b0 = npp_polyval(n, [0, -6, 0, 5, -1]) / s
    r = npp_polyval(n, [0, 0, 2, 1])
    a1 = npp_polyval(n, [-6, -6]) / r
    a2 = 6 / r
    k = np.arange(1, n + 1)
    x = npp_polyval(k, [0, a1, a2])
    y = npp_polyval(k, [b0, b1, b2, b3])
    return x + 1j * y