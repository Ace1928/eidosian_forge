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
class interp1d(_Interpolator1D):
    """
    Interpolate a 1-D function.

    .. legacy:: class

        For a guide to the intended replacements for `interp1d` see
        :ref:`tutorial-interpolate_1Dsection`.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``. This class returns a function whose call method uses
    interpolation to find the value of new points.

    Parameters
    ----------
    x : (npoints, ) array_like
        A 1-D array of real values.
    y : (..., npoints, ...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`. Use the ``axis`` parameter
        to select correct axis. Unlike other interpolators, the default
        interpolation axis is the last axis of `y`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use.
        The string has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
        zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5)
        in that 'nearest-up' rounds up and 'nearest' rounds down. Default
        is 'linear'.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Unlike
        other interpolators, defaults to ``axis=-1``.
    copy : bool, optional
        If True, the class makes internal copies of x and y.
        If False, references to `x` and `y` are used. The default is to copy.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless ``fill_value="extrapolate"``.
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``. Using a two-element tuple
          or ndarray requires ``bounds_error=False``.

          .. versionadded:: 0.17.0
        - If "extrapolate", then points outside the data range will be
          extrapolated.

          .. versionadded:: 0.17.0
    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.

    Attributes
    ----------
    fill_value

    Methods
    -------
    __call__

    See Also
    --------
    splrep, splev
        Spline interpolation/smoothing based on FITPACK.
    UnivariateSpline : An object-oriented wrapper of the FITPACK routines.
    interp2d : 2-D interpolation

    Notes
    -----
    Calling `interp1d` with NaNs present in input values results in
    undefined behaviour.

    Input values `x` and `y` must be convertible to `float` values like
    `int` or `float`.

    If the values in `x` are not unique, the resulting behavior is
    undefined and specific to the choice of `kind`, i.e., changing
    `kind` will change the behavior for duplicates.


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import interpolate
    >>> x = np.arange(0, 10)
    >>> y = np.exp(-x/3.0)
    >>> f = interpolate.interp1d(x, y)

    >>> xnew = np.arange(0, 9, 0.1)
    >>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
    >>> plt.plot(x, y, 'o', xnew, ynew, '-')
    >>> plt.show()
    """

    def __init__(self, x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=False):
        """ Initialize a 1-D linear interpolation class."""
        _Interpolator1D.__init__(self, x, y, axis=axis)
        self.bounds_error = bounds_error
        self.copy = copy
        if kind in ['zero', 'slinear', 'quadratic', 'cubic']:
            order = {'zero': 0, 'slinear': 1, 'quadratic': 2, 'cubic': 3}[kind]
            kind = 'spline'
        elif isinstance(kind, int):
            order = kind
            kind = 'spline'
        elif kind not in ('linear', 'nearest', 'nearest-up', 'previous', 'next'):
            raise NotImplementedError('%s is unsupported: Use fitpack routines for other types.' % kind)
        x = array(x, copy=self.copy)
        y = array(y, copy=self.copy)
        if not assume_sorted:
            ind = np.argsort(x, kind='mergesort')
            x = x[ind]
            y = np.take(y, ind, axis=axis)
        if x.ndim != 1:
            raise ValueError('the x array must have exactly one dimension.')
        if y.ndim == 0:
            raise ValueError('the y array must have at least one dimension.')
        if not issubclass(y.dtype.type, np.inexact):
            y = y.astype(np.float64)
        self.axis = axis % y.ndim
        self.y = y
        self._y = self._reshape_yi(self.y)
        self.x = x
        del y, x
        self._kind = kind
        if kind in ('linear', 'nearest', 'nearest-up', 'previous', 'next'):
            minval = 1
            if kind == 'nearest':
                self._side = 'left'
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]
                self._call = self.__class__._call_nearest
            elif kind == 'nearest-up':
                self._side = 'right'
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]
                self._call = self.__class__._call_nearest
            elif kind == 'previous':
                self._side = 'left'
                self._ind = 0
                self._x_shift = np.nextafter(self.x, -np.inf)
                self._call = self.__class__._call_previousnext
                if _do_extrapolate(fill_value):
                    self._check_and_update_bounds_error_for_extrapolation()
                    fill_value = (np.nan, np.take(self.y, -1, axis))
            elif kind == 'next':
                self._side = 'right'
                self._ind = 1
                self._x_shift = np.nextafter(self.x, np.inf)
                self._call = self.__class__._call_previousnext
                if _do_extrapolate(fill_value):
                    self._check_and_update_bounds_error_for_extrapolation()
                    fill_value = (np.take(self.y, 0, axis), np.nan)
            else:
                np_dtypes = (np.dtype(np.float64), np.dtype(int))
                cond = self.x.dtype in np_dtypes and self.y.dtype in np_dtypes
                cond = cond and self.y.ndim == 1
                cond = cond and (not _do_extrapolate(fill_value))
                if cond:
                    self._call = self.__class__._call_linear_np
                else:
                    self._call = self.__class__._call_linear
        else:
            minval = order + 1
            rewrite_nan = False
            xx, yy = (self.x, self._y)
            if order > 1:
                mask = np.isnan(self.x)
                if mask.any():
                    sx = self.x[~mask]
                    if sx.size == 0:
                        raise ValueError('`x` array is all-nan')
                    xx = np.linspace(np.nanmin(self.x), np.nanmax(self.x), len(self.x))
                    rewrite_nan = True
                if np.isnan(self._y).any():
                    yy = np.ones_like(self._y)
                    rewrite_nan = True
            self._spline = make_interp_spline(xx, yy, k=order, check_finite=False)
            if rewrite_nan:
                self._call = self.__class__._call_nan_spline
            else:
                self._call = self.__class__._call_spline
        if len(self.x) < minval:
            raise ValueError('x and y arrays must have at least %d entries' % minval)
        self.fill_value = fill_value

    @property
    def fill_value(self):
        """The fill value."""
        return self._fill_value_orig

    @fill_value.setter
    def fill_value(self, fill_value):
        if _do_extrapolate(fill_value):
            self._check_and_update_bounds_error_for_extrapolation()
            self._extrapolate = True
        else:
            broadcast_shape = self.y.shape[:self.axis] + self.y.shape[self.axis + 1:]
            if len(broadcast_shape) == 0:
                broadcast_shape = (1,)
            if isinstance(fill_value, tuple) and len(fill_value) == 2:
                below_above = [np.asarray(fill_value[0]), np.asarray(fill_value[1])]
                names = ('fill_value (below)', 'fill_value (above)')
                for ii in range(2):
                    below_above[ii] = _check_broadcast_up_to(below_above[ii], broadcast_shape, names[ii])
            else:
                fill_value = np.asarray(fill_value)
                below_above = [_check_broadcast_up_to(fill_value, broadcast_shape, 'fill_value')] * 2
            self._fill_value_below, self._fill_value_above = below_above
            self._extrapolate = False
            if self.bounds_error is None:
                self.bounds_error = True
        self._fill_value_orig = fill_value

    def _check_and_update_bounds_error_for_extrapolation(self):
        if self.bounds_error:
            raise ValueError('Cannot extrapolate and raise at the same time.')
        self.bounds_error = False

    def _call_linear_np(self, x_new):
        return np.interp(x_new, self.x, self.y)

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

    def _call_nearest(self, x_new):
        """ Find nearest neighbor interpolated y_new = f(x_new)."""
        x_new_indices = searchsorted(self.x_bds, x_new, side=self._side)
        x_new_indices = x_new_indices.clip(0, len(self.x) - 1).astype(intp)
        y_new = self._y[x_new_indices]
        return y_new

    def _call_previousnext(self, x_new):
        """Use previous/next neighbor of x_new, y_new = f(x_new)."""
        x_new_indices = searchsorted(self._x_shift, x_new, side=self._side)
        x_new_indices = x_new_indices.clip(1 - self._ind, len(self.x) - self._ind).astype(intp)
        y_new = self._y[x_new_indices + self._ind - 1]
        return y_new

    def _call_spline(self, x_new):
        return self._spline(x_new)

    def _call_nan_spline(self, x_new):
        out = self._spline(x_new)
        out[...] = np.nan
        return out

    def _evaluate(self, x_new):
        x_new = asarray(x_new)
        y_new = self._call(self, x_new)
        if not self._extrapolate:
            below_bounds, above_bounds = self._check_bounds(x_new)
            if len(y_new) > 0:
                y_new[below_bounds] = self._fill_value_below
                y_new[above_bounds] = self._fill_value_above
        return y_new

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