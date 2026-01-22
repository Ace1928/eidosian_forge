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
def _bessel_poly(n, reverse=False):
    """
    Return the coefficients of Bessel polynomial of degree `n`

    If `reverse` is true, a reverse Bessel polynomial is output.

    Output is a list of coefficients:
    [1]                   = 1
    [1,  1]               = 1*s   +  1
    [1,  3,  3]           = 1*s^2 +  3*s   +  3
    [1,  6, 15, 15]       = 1*s^3 +  6*s^2 + 15*s   +  15
    [1, 10, 45, 105, 105] = 1*s^4 + 10*s^3 + 45*s^2 + 105*s + 105
    etc.

    Output is a Python list of arbitrary precision long ints, so n is only
    limited by your hardware's memory.

    Sequence is http://oeis.org/A001498, and output can be confirmed to
    match http://oeis.org/A001498/b001498.txt :

    >>> from scipy.signal._filter_design import _bessel_poly
    >>> i = 0
    >>> for n in range(51):
    ...     for x in _bessel_poly(n, reverse=True):
    ...         print(i, x)
    ...         i += 1

    """
    if abs(int(n)) != n:
        raise ValueError('Polynomial order must be a nonnegative integer')
    else:
        n = int(n)
    out = []
    for k in range(n + 1):
        num = _falling_factorial(2 * n - k, n)
        den = 2 ** (n - k) * math.factorial(k)
        out.append(num // den)
    if reverse:
        return out[::-1]
    else:
        return out