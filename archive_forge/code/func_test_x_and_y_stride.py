from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_x_and_y_stride(self):
    x = arange(12.0, dtype=self.dtype)
    y = zeros(6, x.dtype)
    desired_x = y.copy()[::2]
    desired_y = x.copy()[::4]
    x, y = self.blas_func(x, y, n=3, incx=4, incy=2)
    assert_array_equal(desired_x, x[::4])
    assert_array_equal(desired_y, y[::2])