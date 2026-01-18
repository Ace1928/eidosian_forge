import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_perm_and_scaling(self):
    cases = (np.array([[0.0, 0.0, 0.0, 0.0, 2e-06], [0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2e-06, 0.0, 0.0]]), np.array([[-0.5, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [1.0, 0.0, -0.5, 0.0], [0.0, 1.0, 0.0, -1.0]]), np.array([[-3.0, 0.0, 1.0, 0.0], [-1.0, -1.0, -0.0, 1.0], [-3.0, -0.0, -0.0, 0.0], [-1.0, -0.0, 1.0, -1.0]]))
    for A in cases:
        x, y = matrix_balance(A)
        x, (s, p) = matrix_balance(A, separate=1)
        ip = np.empty_like(p)
        ip[p] = np.arange(A.shape[0])
        assert_allclose(y, np.diag(s)[ip, :])
        assert_allclose(solve(y, A).dot(y), x)