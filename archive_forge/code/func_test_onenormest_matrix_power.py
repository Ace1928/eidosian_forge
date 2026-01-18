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
def test_onenormest_matrix_power(self):
    np.random.seed(1234)
    n = 40
    nsamples = 10
    for i in range(nsamples):
        A = scipy.linalg.inv(np.random.randn(n, n))
        for p in range(4):
            if not p:
                M = np.identity(n)
            else:
                M = np.dot(M, A)
            estimated = _onenormest_matrix_power(A, p)
            exact = np.linalg.norm(M, 1)
            assert_(less_than_or_close(estimated, exact))
            assert_(less_than_or_close(exact, 3 * estimated))