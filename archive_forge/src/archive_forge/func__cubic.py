import warnings
from numpy import (logical_and, asarray, pi, zeros_like,
from numpy import (sqrt, exp, greater, less, cos, add, sin, less_equal,
from ._spline import cspline2d, sepfir2d
from ._signaltools import lfilter, sosfilt, lfiltic
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.interpolate import BSpline
def _cubic(x):
    x = asarray(x, dtype=float)
    b = BSpline.basis_element([-2, -1, 0, 1, 2], extrapolate=False)
    out = b(x)
    out[(x < -2) | (x > 2)] = 0
    return out