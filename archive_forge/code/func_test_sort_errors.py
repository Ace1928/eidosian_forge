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
def test_sort_errors(self):
    a = [[4.0, 3.0, 1.0, -1.0], [-4.5, -3.5, -1.0, 1.0], [9.0, 6.0, -4.0, 4.5], [6.0, 4.0, -3.0, 3.5]]
    assert_raises(ValueError, schur, a, sort='unsupported')
    assert_raises(ValueError, schur, a, sort=1)