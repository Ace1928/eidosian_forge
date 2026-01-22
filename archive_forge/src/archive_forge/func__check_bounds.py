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
def _check_bounds(self, x_new):
    """Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array

        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """
    below_bounds = x_new < self.x[0]
    above_bounds = x_new > self.x[-1]
    if self.bounds_error and below_bounds.any():
        below_bounds_value = x_new[np.argmax(below_bounds)]
        raise ValueError(f"A value ({below_bounds_value}) in x_new is below the interpolation range's minimum value ({self.x[0]}).")
    if self.bounds_error and above_bounds.any():
        above_bounds_value = x_new[np.argmax(above_bounds)]
        raise ValueError(f"A value ({above_bounds_value}) in x_new is above the interpolation range's maximum value ({self.x[-1]}).")
    return (below_bounds, above_bounds)