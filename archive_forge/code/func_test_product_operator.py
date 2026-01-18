import math
import numpy as np
from numpy import array, eye, exp, random
from numpy.testing import (
from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning
from scipy.sparse._construct import eye as speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,
from scipy.sparse._sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg
def test_product_operator(self):
    random.seed(1234)
    n = 5
    k = 2
    nsamples = 10
    for i in range(nsamples):
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        C = np.random.randn(n, n)
        D = np.random.randn(n, k)
        op = ProductOperator(A, B, C)
        assert_allclose(op.matmat(D), A.dot(B).dot(C).dot(D))
        assert_allclose(op.T.matmat(D), A.dot(B).dot(C).T.dot(D))