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
@pytest.mark.parametrize('dtype_', DTYPES)
@pytest.mark.parametrize('driver', ('ev', 'evd', 'evr', 'evx'))
def test_various_drivers_standard(self, driver, dtype_):
    a = _random_hermitian_matrix(n=20, dtype=dtype_)
    w, v = eigh(a, driver=driver)
    assert_allclose(a @ v - v * w, 0.0, atol=1000 * np.finfo(dtype_).eps, rtol=0.0)