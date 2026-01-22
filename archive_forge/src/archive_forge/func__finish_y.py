import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
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