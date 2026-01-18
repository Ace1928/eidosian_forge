import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_vector_scalar(self):
    x0 = np.array([100.0, -0.5])
    jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0, method='2-point', as_linear_operator=True)
    jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0, as_linear_operator=True)
    jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0, method='cs', as_linear_operator=True)
    jac_true = self.jac_vector_scalar(x0)
    np.random.seed(1)
    for i in range(10):
        p = np.random.uniform(-10, 10, size=x0.shape)
        assert_allclose(jac_diff_2.dot(p), np.atleast_1d(jac_true.dot(p)), rtol=1e-05)
        assert_allclose(jac_diff_3.dot(p), np.atleast_1d(jac_true.dot(p)), rtol=5e-06)
        assert_allclose(jac_diff_4.dot(p), np.atleast_1d(jac_true.dot(p)), rtol=1e-07)