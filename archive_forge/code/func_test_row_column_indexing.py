import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_row_column_indexing(self):
    x = asmatrix(np.eye(2))
    assert_array_equal(x[0, :], [[1, 0]])
    assert_array_equal(x[1, :], [[0, 1]])
    assert_array_equal(x[:, 0], [[1], [0]])
    assert_array_equal(x[:, 1], [[0], [1]])