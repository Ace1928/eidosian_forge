from functools import partial
from itertools import product
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
from scipy._lib._util import np_long
def test_sparse_expm_multiply(self):
    np.random.seed(1234)
    n = 40
    k = 3
    nsamples = 10
    for i in range(nsamples):
        A = scipy.sparse.rand(n, n, density=0.05)
        B = np.random.randn(n, k)
        observed = expm_multiply(A, B)
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'splu converted its input to CSC format')
            sup.filter(SparseEfficiencyWarning, 'spsolve is more efficient when sparse b is in the CSC matrix format')
            expected = sp_expm(A).dot(B)
        assert_allclose(observed, expected)
        observed = estimated(expm_multiply)(aslinearoperator(A), B)
        assert_allclose(observed, expected)