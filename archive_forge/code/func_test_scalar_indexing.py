import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_scalar_indexing(self):
    x = asmatrix(np.zeros((3, 2), float))
    assert_equal(x[0, 0], x[0][0])