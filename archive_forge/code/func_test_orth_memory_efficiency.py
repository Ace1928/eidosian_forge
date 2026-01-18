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
@pytest.mark.slow
@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8, reason='test only on 64-bit, else too slow')
def test_orth_memory_efficiency():
    n = 10 * 1000 * 1000
    try:
        _check_orth(n, np.float64, skip_big=True)
    except MemoryError as e:
        raise AssertionError('memory error perhaps caused by orth regression') from e