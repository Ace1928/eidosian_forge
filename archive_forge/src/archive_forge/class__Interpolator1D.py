import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
class _Interpolator1D:
    """Common features in univariate interpolation.

    Deal with input data type and interpolation axis rolling. The
    actual interpolator can assume the y-data is of shape (n, r) where
    `n` is the number of x-points, and `r` the number of variables,
    and use self.dtype as the y-data type.

    Attributes
    ----------
    _y_axis : Axis along which the interpolation goes in the
        original array
    _y_extra_shape : Additional shape of the input arrays, excluding
        the interpolation axis
    dtype : Dtype of the y-data arrays. It can be set via _set_dtype,
        which forces it to be float or complex

    Methods
    -------
    __call__
    _prepare_x
    _finish_y
    _reshape_y
    _reshape_yi
    _set_yi
    _set_dtype
    _evaluate

    """

    def __init__(self, xi=None, yi=None, axis=None):
        self._y_axis = axis
        self._y_extra_shape = None
        self.dtype = None
        if yi is not None:
            self._set_yi(yi, xi=xi, axis=axis)

    def __call__(self, x):
        """Evaluate the interpolant

        Parameters
        ----------
        x : cupy.ndarray
            The points to evaluate the interpolant

        Returns
        -------
        y : cupy.ndarray
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x

        Notes
        -----
        Input values `x` must be convertible to `float` values like `int`
        or `float`.

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate(x)
        return self._finish_y(y, x_shape)

    def _evaluate(self, x):
        """
        Actually evaluate the value of the interpolator
        """
        raise NotImplementedError()

    def _prepare_x(self, x):
        """
        Reshape input array to 1-D
        """
        x = _asarray_validated(x, check_finite=False, as_inexact=True)
        x_shape = x.shape
        return (x.ravel(), x_shape)

    def _finish_y(self, y, x_shape):
        """
        Reshape interpolated y back to an N-D array similar to initial y
        """
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = list(range(nx, nx + self._y_axis)) + list(range(nx)) + list(range(nx + self._y_axis, nx + ny))
            y = y.transpose(s)
        return y

    def _reshape_yi(self, yi, check=False):
        """
        Reshape the updated yi to a 1-D array
        """
        yi = cupy.moveaxis(yi, self._y_axis, 0)
        if check and yi.shape[1:] != self._y_extra_shape:
            ok_shape = '%r + (N,) + %r' % (self._y_extra_shape[-self._y_axis:], self._y_extra_shape[:-self._y_axis])
            raise ValueError('Data must be of shape %s' % ok_shape)
        return yi.reshape((yi.shape[0], -1))

    def _set_yi(self, yi, xi=None, axis=None):
        if axis is None:
            axis = self._y_axis
        if axis is None:
            raise ValueError('no interpolation axis specified')
        shape = yi.shape
        if shape == ():
            shape = (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError('x and y arrays must be equal in length along interpolation axis.')
        self._y_axis = axis % yi.ndim
        self._y_extra_shape = yi.shape[:self._y_axis] + yi.shape[self._y_axis + 1:]
        self.dtype = None
        self._set_dtype(yi.dtype)

    def _set_dtype(self, dtype, union=False):
        if cupy.issubdtype(dtype, cupy.complexfloating) or cupy.issubdtype(self.dtype, cupy.complexfloating):
            self.dtype = cupy.complex_
        elif not union or self.dtype != cupy.complex_:
            self.dtype = cupy.float_