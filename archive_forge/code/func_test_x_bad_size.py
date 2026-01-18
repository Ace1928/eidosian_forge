from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_x_bad_size(self):
    x = arange(12.0, dtype=self.dtype)
    y = zeros(6, x.dtype)
    with pytest.raises(Exception, match='failed for 1st keyword'):
        self.blas_func(x, y, n=4, incx=5)