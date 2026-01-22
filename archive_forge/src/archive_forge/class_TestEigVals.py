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
class TestEigVals:

    def test_simple(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        w = eigvals(a)
        exact_w = [(9 + sqrt(93)) / 2, 0, (9 - sqrt(93)) / 2]
        assert_array_almost_equal(w, exact_w)

    def test_simple_tr(self):
        a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6]], 'd').T
        a = a.copy()
        a = a.T
        w = eigvals(a)
        exact_w = [(9 + sqrt(93)) / 2, 0, (9 - sqrt(93)) / 2]
        assert_array_almost_equal(w, exact_w)

    def test_simple_complex(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6 + 1j]]
        w = eigvals(a)
        exact_w = [(9 + 1j + sqrt(92 + 6j)) / 2, 0, (9 + 1j - sqrt(92 + 6j)) / 2]
        assert_array_almost_equal(w, exact_w)

    def test_finite(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        w = eigvals(a, check_finite=False)
        exact_w = [(9 + sqrt(93)) / 2, 0, (9 - sqrt(93)) / 2]
        assert_array_almost_equal(w, exact_w)