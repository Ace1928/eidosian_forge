from numpy.testing import (assert_, assert_allclose, assert_equal,
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import gcrotmk, gmres
def test_truncate(self):
    np.random.seed(1234)
    A = np.random.rand(30, 30) + np.eye(30)
    b = np.random.rand(30)
    for truncate in ['oldest', 'smallest']:
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, '.*called without specifying.*')
            x, info = gcrotmk(A, b, m=10, k=10, truncate=truncate, rtol=0.0001, maxiter=200)
        assert_equal(info, 0)
        assert_allclose(A.dot(x) - b, 0, atol=0.001)