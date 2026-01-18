import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_numpy_ravel(self):
    assert_equal(np.ravel(self.a).shape, (2,))
    assert_equal(np.ravel(self.m).shape, (2,))