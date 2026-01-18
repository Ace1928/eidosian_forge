import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
def test_random_complex_economic(self):
    rng = np.random.RandomState(1234)
    m = 100
    n = 200
    for k in range(2):
        a = rng.random([m, n]) + 1j * rng.random([m, n])
        r, q = rq(a, mode='economic')
        assert_array_almost_equal(q @ q.conj().T, eye(m))
        assert_array_almost_equal(r @ q, a)
        assert_equal(q.shape, (m, n))
        assert_equal(r.shape, (m, m))