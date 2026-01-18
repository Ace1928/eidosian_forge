import os
import pytest
import sys
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.linalg._svdp import _svdp
from scipy.sparse import csr_matrix, csc_matrix
@pytest.mark.slow
@pytest.mark.xfail()
def test_shifts_accuracy():
    np.random.seed(0)
    n, k = (70, 10)
    A = np.random.random((n, n)).astype(np.float64)
    u1, s1, vt1, _ = _svdp(A, k, shifts=None, which='SM', irl_mode=True)
    u2, s2, vt2, _ = _svdp(A, k, shifts=32, which='SM', irl_mode=True)
    assert_allclose(s1, s2)