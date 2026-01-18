from functools import partial
from itertools import product
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
from scipy._lib._util import np_long
def test_sparse_expm_multiply_interval_dtypes(self):
    A = scipy.sparse.diags(np.arange(5), format='csr', dtype=int)
    B = np.ones(5, dtype=int)
    Aexpm = scipy.sparse.diags(np.exp(np.arange(5)), format='csr')
    assert_allclose(expm_multiply(A, B, 0, 1)[-1], Aexpm.dot(B))
    A = scipy.sparse.diags(-1j * np.arange(5), format='csr', dtype=complex)
    B = np.ones(5, dtype=int)
    Aexpm = scipy.sparse.diags(np.exp(-1j * np.arange(5)), format='csr')
    assert_allclose(expm_multiply(A, B, 0, 1)[-1], Aexpm.dot(B))
    A = scipy.sparse.diags(np.arange(5), format='csr', dtype=int)
    B = np.full(5, 1j, dtype=complex)
    Aexpm = scipy.sparse.diags(np.exp(np.arange(5)), format='csr')
    assert_allclose(expm_multiply(A, B, 0, 1)[-1], Aexpm.dot(B))