import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_returntype(self):
    a = np.array([[0, 1], [0, 0]])
    assert_(type(matrix_power(a, 2)) is np.ndarray)
    a = mat(a)
    assert_(type(matrix_power(a, 2)) is matrix)