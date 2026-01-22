import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def _prepare_x(self, x):
    """
        Reshape input array to 1-D
        """
    x = _asarray_validated(x, check_finite=False, as_inexact=True)
    x_shape = x.shape
    return (x.ravel(), x_shape)