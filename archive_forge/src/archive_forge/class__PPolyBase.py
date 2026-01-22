import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
class _PPolyBase:
    """Base class for piecewise polynomials."""
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        self.c = cupy.asarray(c)
        self.x = cupy.ascontiguousarray(x, dtype=cupy.float64)
        if extrapolate is None:
            extrapolate = True
        elif extrapolate != 'periodic':
            extrapolate = bool(extrapolate)
        self.extrapolate = extrapolate
        if self.c.ndim < 2:
            raise ValueError('Coefficients array must be at least 2-dimensional.')
        if not 0 <= axis < self.c.ndim - 1:
            raise ValueError('axis=%s must be between 0 and %s' % (axis, self.c.ndim - 1))
        self.axis = axis
        if axis != 0:
            self.c = cupy.moveaxis(self.c, axis + 1, 0)
            self.c = cupy.moveaxis(self.c, axis + 1, 0)
        if self.x.ndim != 1:
            raise ValueError('x must be 1-dimensional')
        if self.x.size < 2:
            raise ValueError('at least 2 breakpoints are needed')
        if self.c.ndim < 2:
            raise ValueError('c must have at least 2 dimensions')
        if self.c.shape[0] == 0:
            raise ValueError('polynomial must be at least of order 0')
        if self.c.shape[1] != self.x.size - 1:
            raise ValueError('number of coefficients != len(x)-1')
        dx = cupy.diff(self.x)
        if not (cupy.all(dx >= 0) or cupy.all(dx <= 0)):
            raise ValueError('`x` must be strictly increasing or decreasing.')
        dtype = self._get_dtype(self.c.dtype)
        self.c = cupy.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if cupy.issubdtype(dtype, cupy.complexfloating) or cupy.issubdtype(self.c.dtype, cupy.complexfloating):
            return cupy.complex_
        else:
            return cupy.float_

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        """
        Construct the piecewise polynomial without making checks.
        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type. The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.
        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _ensure_c_contiguous(self):
        """
        c and x may be modified by the user. The Cython code expects
        that they are C contiguous.
        """
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x):
        """
        Add additional breakpoints and coefficients to the polynomial.

        Parameters
        ----------
        c : ndarray, size (k, m, ...)
            Additional coefficients for polynomials in intervals. Note that
            the first additional interval will be formed using one of the
            ``self.x`` end points.
        x : ndarray, size (m,)
            Additional breakpoints. Must be sorted in the same order as
            ``self.x`` and either to the right or to the left of the current
            breakpoints.
        """
        c = cupy.asarray(c)
        x = cupy.asarray(x)
        if c.ndim < 2:
            raise ValueError('invalid dimensions for c')
        if x.ndim != 1:
            raise ValueError('invalid dimensions for x')
        if x.shape[0] != c.shape[1]:
            raise ValueError('Shapes of x {} and c {} are incompatible'.format(x.shape, c.shape))
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError('Shapes of c {} and self.c {} are incompatible'.format(c.shape, self.c.shape))
        if c.size == 0:
            return
        dx = cupy.diff(x)
        if not (cupy.all(dx >= 0) or cupy.all(dx <= 0)):
            raise ValueError('`x` is not sorted.')
        if self.x[-1] >= self.x[0]:
            if not x[-1] >= x[0]:
                raise ValueError('`x` is in the different order than `self.x`.')
            if x[0] >= self.x[-1]:
                action = 'append'
            elif x[-1] <= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError('`x` is neither on the left or on the right from `self.x`.')
        else:
            if not x[-1] <= x[0]:
                raise ValueError('`x` is in the different order than `self.x`.')
            if x[0] <= self.x[-1]:
                action = 'append'
            elif x[-1] >= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError('`x` is neither on the left or on the right from `self.x`.')
        dtype = self._get_dtype(c.dtype)
        k2 = max(c.shape[0], self.c.shape[0])
        c2 = cupy.zeros((k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:], dtype=dtype)
        if action == 'append':
            c2[k2 - self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2 - c.shape[0]:, self.c.shape[1]:] = c
            self.x = cupy.r_[self.x, x]
        elif action == 'prepend':
            c2[k2 - self.c.shape[0]:, :c.shape[1]] = c
            c2[k2 - c.shape[0]:, c.shape[1]:] = self.c
            self.x = cupy.r_[x, self.x]
        self.c = c2

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative.

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = cupy.asarray(x)
        x_shape, x_ndim = (x.shape, x.ndim)
        x = cupy.ascontiguousarray(x.ravel(), dtype=cupy.float_)
        if extrapolate == 'periodic':
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False
        out = cupy.empty((len(x), int(np.prod(self.c.shape[2:]))), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            dims = list(range(out.ndim))
            dims = dims[x_ndim:x_ndim + self.axis] + dims[:x_ndim] + dims[x_ndim + self.axis:]
            out = out.transpose(dims)
        return out