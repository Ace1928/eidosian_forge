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
def test_random_complex_left(self):
    rng = np.random.RandomState(1234)
    n = 20
    for k in range(2):
        a = rng.random([n, n]) + 1j * rng.random([n, n])
        q, r = qr(a)
        c = rng.random([n]) + 1j * rng.random([n])
        qc, r = qr_multiply(a, c, 'left')
        assert_array_almost_equal(q @ c, qc)
        qc, r = qr_multiply(a, eye(n), 'left')
        assert_array_almost_equal(q, qc)