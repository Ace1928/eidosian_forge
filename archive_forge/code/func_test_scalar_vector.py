import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_scalar_vector(self):
    x0 = 0.5
    jac_diff_2 = approx_derivative(self.fun_scalar_vector, x0, method='2-point', as_linear_operator=True)
    jac_diff_3 = approx_derivative(self.fun_scalar_vector, x0, as_linear_operator=True)
    jac_diff_4 = approx_derivative(self.fun_scalar_vector, x0, method='cs', as_linear_operator=True)
    jac_true = self.jac_scalar_vector(np.atleast_1d(x0))
    np.random.seed(1)
    for i in range(10):
        p = np.random.uniform(-10, 10, size=(1,))
        assert_allclose(jac_diff_2.dot(p), jac_true.dot(p), rtol=1e-05)
        assert_allclose(jac_diff_3.dot(p), jac_true.dot(p), rtol=5e-06)
        assert_allclose(jac_diff_4.dot(p), jac_true.dot(p), rtol=5e-06)