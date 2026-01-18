import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
def test_syequb():
    desired_log2s = np.array([0, 0, 0, 0, 0, 0, -1, -1, -2, -3])
    for ind, dtype in enumerate(DTYPES):
        A = np.eye(10, dtype=dtype)
        alpha = dtype(1.0 if ind < 2 else 1j)
        d = np.array([alpha * 2.0 ** x for x in range(-5, 5)], dtype=dtype)
        A += np.rot90(np.diag(d))
        syequb = get_lapack_funcs('syequb', dtype=dtype)
        s, scond, amax, info = syequb(A)
        assert_equal(np.log2(s).astype(int), desired_log2s)