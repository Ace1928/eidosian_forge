from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_x_stride_transpose(self):
    alpha, beta, a, x, y = self.get_data(x_stride=2)
    desired_y = alpha * matrixmultiply(transpose(a), x[::2]) + beta * y
    y = self.blas_func(alpha, a, x, beta, y, trans=1, incx=2)
    assert_array_almost_equal(desired_y, y)