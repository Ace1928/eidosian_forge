from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_simple_transpose_conj(self):
    alpha, beta, a, x, y = self.get_data()
    desired_y = alpha * matrixmultiply(transpose(conjugate(a)), x) + beta * y
    y = self.blas_func(alpha, a, x, beta, y, trans=2)
    assert_array_almost_equal(desired_y, y)