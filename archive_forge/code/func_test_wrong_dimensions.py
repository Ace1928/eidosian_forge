import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_wrong_dimensions(self):
    x0 = 1.0
    assert_raises(RuntimeError, approx_derivative, self.wrong_dimensions_fun, x0)
    f0 = self.wrong_dimensions_fun(np.atleast_1d(x0))
    assert_raises(ValueError, approx_derivative, self.wrong_dimensions_fun, x0, f0=f0)