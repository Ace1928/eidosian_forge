import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
def test_real_eigs_real_k_subset():
    np.random.seed(1)
    n = 10
    A = rand(n, n, density=0.5)
    A.data *= 2
    A.data -= 1
    v0 = np.ones(n)
    whichs = ['LM', 'SM', 'LR', 'SR', 'LI', 'SI']
    dtypes = [np.float32, np.float64]
    for which, sigma, dtype in itertools.product(whichs, [None, 0, 5], dtypes):
        prev_w = np.array([], dtype=dtype)
        eps = np.finfo(dtype).eps
        for k in range(1, 9):
            w, z = eigs(A.astype(dtype), k=k, which=which, sigma=sigma, v0=v0.astype(dtype), tol=0)
            assert_allclose(np.linalg.norm(A.dot(z) - z * w), 0, atol=np.sqrt(eps))
            dist = abs(prev_w[:, None] - w).min(axis=1)
            assert_allclose(dist, 0, atol=np.sqrt(eps))
            prev_w = w
            if sigma is None:
                d = w
            else:
                d = 1 / (w - sigma)
            if which == 'LM':
                assert np.all(np.diff(abs(d)) <= 1e-06)