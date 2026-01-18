import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_with_bounds_2_point(self):
    lb = -np.ones(2)
    ub = np.ones(2)
    x0 = np.array([-2.0, 0.2])
    assert_raises(ValueError, approx_derivative, self.fun_vector_vector, x0, bounds=(lb, ub))
    x0 = np.array([-1.0, 1.0])
    jac_diff = approx_derivative(self.fun_vector_vector, x0, method='2-point', bounds=(lb, ub))
    jac_true = self.jac_vector_vector(x0)
    assert_allclose(jac_diff, jac_true, rtol=1e-06)