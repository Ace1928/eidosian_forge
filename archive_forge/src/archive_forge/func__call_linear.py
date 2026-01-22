from math import prod
import warnings
import numpy as np
from numpy import (array, transpose, searchsorted, atleast_1d, atleast_2d,
import scipy.special as spec
from scipy.special import comb
from . import _fitpack_py
from . import dfitpack
from ._polyint import _Interpolator1D
from . import _ppoly
from .interpnd import _ndim_coords_from_arrays
from ._bsplines import make_interp_spline, BSpline
import itertools  # noqa: F401
def _call_linear(self, x_new):
    x_new_indices = searchsorted(self.x, x_new)
    x_new_indices = x_new_indices.clip(1, len(self.x) - 1).astype(int)
    lo = x_new_indices - 1
    hi = x_new_indices
    x_lo = self.x[lo]
    x_hi = self.x[hi]
    y_lo = self._y[lo]
    y_hi = self._y[hi]
    slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]
    y_new = slope * (x_new - x_lo)[:, None] + y_lo
    return y_new