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
def test_ticket_1645(self):
    for dtype in DTYPES:
        a = np.zeros((300, 2), dtype=dtype)
        gerqf, = get_lapack_funcs(['gerqf'], [a])
        assert_raises(Exception, gerqf, a, lwork=2)
        rq, tau, work, info = gerqf(a)
        if dtype in REAL_DTYPES:
            orgrq, = get_lapack_funcs(['orgrq'], [a])
            assert_raises(Exception, orgrq, rq[-2:], tau, lwork=1)
            orgrq(rq[-2:], tau, lwork=2)
        elif dtype in COMPLEX_DTYPES:
            ungrq, = get_lapack_funcs(['ungrq'], [a])
            assert_raises(Exception, ungrq, rq[-2:], tau, lwork=1)
            ungrq(rq[-2:], tau, lwork=2)