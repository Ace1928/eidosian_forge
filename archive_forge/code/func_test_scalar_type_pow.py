import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_scalar_type_pow(self):
    m = matrix([[1, 2], [3, 4]])
    for scalar_t in [np.int8, np.uint8]:
        two = scalar_t(2)
        assert_array_almost_equal(m ** 2, m ** two)