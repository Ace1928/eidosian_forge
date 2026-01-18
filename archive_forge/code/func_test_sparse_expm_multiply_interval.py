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
def test_sparse_expm_multiply_interval(self):
    np.random.seed(1234)
    start = 0.1
    stop = 3.2
    n = 40
    k = 3
    endpoint = True
    for num in (14, 13, 2):
        A = scipy.sparse.rand(n, n, density=0.05)
        B = np.random.randn(n, k)
        v = np.random.randn(n)
        for target in (B, v):
            X = expm_multiply(A, target, start=start, stop=stop, num=num, endpoint=endpoint)
            samples = np.linspace(start=start, stop=stop, num=num, endpoint=endpoint)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'splu converted its input to CSC format')
                sup.filter(SparseEfficiencyWarning, 'spsolve is more efficient when sparse b is in the CSC matrix format')
                for solution, t in zip(X, samples):
                    assert_allclose(solution, sp_expm(t * A).dot(target))