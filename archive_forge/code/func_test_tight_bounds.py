import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_tight_bounds(self):
    x0 = np.array([10.0, 10.0])
    lb = x0 - 3e-09
    ub = x0 + 2e-09
    jac_true = self.jac_vector_vector(x0)
    jac_diff = approx_derivative(self.fun_vector_vector, x0, method='2-point', bounds=(lb, ub))
    assert_allclose(jac_diff, jac_true, rtol=1e-06)
    jac_diff = approx_derivative(self.fun_vector_vector, x0, method='2-point', rel_step=1e-06, bounds=(lb, ub))
    assert_allclose(jac_diff, jac_true, rtol=1e-06)
    jac_diff = approx_derivative(self.fun_vector_vector, x0, bounds=(lb, ub))
    assert_allclose(jac_diff, jac_true, rtol=1e-06)
    jac_diff = approx_derivative(self.fun_vector_vector, x0, rel_step=1e-06, bounds=(lb, ub))
    assert_allclose(jac_true, jac_diff, rtol=1e-06)