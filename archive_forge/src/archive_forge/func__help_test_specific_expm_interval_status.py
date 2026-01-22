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
def _help_test_specific_expm_interval_status(self, target_status):
    np.random.seed(1234)
    start = 0.1
    stop = 3.2
    num = 13
    endpoint = True
    n = 5
    k = 2
    nrepeats = 10
    nsuccesses = 0
    for num in [14, 13, 2] * nrepeats:
        A = np.random.randn(n, n)
        B = np.random.randn(n, k)
        status = _expm_multiply_interval(A, B, start=start, stop=stop, num=num, endpoint=endpoint, status_only=True)
        if status == target_status:
            X, status = _expm_multiply_interval(A, B, start=start, stop=stop, num=num, endpoint=endpoint, status_only=False)
            assert_equal(X.shape, (num, n, k))
            samples = np.linspace(start=start, stop=stop, num=num, endpoint=endpoint)
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t * A).dot(B))
            nsuccesses += 1
    if not nsuccesses:
        msg = 'failed to find a status-' + str(target_status) + ' interval'
        raise Exception(msg)