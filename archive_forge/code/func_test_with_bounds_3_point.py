import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_with_bounds_3_point(self):
    lb = np.array([1.0, 1.0])
    ub = np.array([2.0, 2.0])
    x0 = np.array([1.0, 2.0])
    jac_true = self.jac_vector_vector(x0)
    jac_diff = approx_derivative(self.fun_vector_vector, x0)
    assert_allclose(jac_diff, jac_true, rtol=1e-09)
    jac_diff = approx_derivative(self.fun_vector_vector, x0, bounds=(lb, np.inf))
    assert_allclose(jac_diff, jac_true, rtol=1e-09)
    jac_diff = approx_derivative(self.fun_vector_vector, x0, bounds=(-np.inf, ub))
    assert_allclose(jac_diff, jac_true, rtol=1e-09)
    jac_diff = approx_derivative(self.fun_vector_vector, x0, bounds=(lb, ub))
    assert_allclose(jac_diff, jac_true, rtol=1e-09)