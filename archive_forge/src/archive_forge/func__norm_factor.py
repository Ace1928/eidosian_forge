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
def _norm_factor(p, k):
    """
    Numerically find frequency shift to apply to delay-normalized filter such
    that -3 dB point is at 1 rad/sec.

    `p` is an array_like of polynomial poles
    `k` is a float gain

    First 10 values are listed in "Bessel Scale Factors" table,
    "Bessel Filters Polynomials, Poles and Circuit Elements 2003, C. Bond."
    """
    p = asarray(p, dtype=complex)

    def G(w):
        """
        Gain of filter
        """
        return abs(k / prod(1j * w - p))

    def cutoff(w):
        """
        When gain = -3 dB, return 0
        """
        return G(w) - 1 / np.sqrt(2)
    return optimize.newton(cutoff, 1.5)