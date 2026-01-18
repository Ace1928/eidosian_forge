import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
def test_zero_diagonal(self):
    n = 5
    rng = np.random.default_rng(43876432987)
    A = rng.standard_normal((n, n))
    b = np.arange(n)
    A = scipy.sparse.tril(A, k=0, format='csr')
    x = spsolve_triangular(A, b, unit_diagonal=True, lower=True)
    A.setdiag(1)
    assert_allclose(A.dot(x), b)
    A = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0])
    with suppress_warnings() as sup:
        sup.filter(SparseEfficiencyWarning, 'CSR matrix format is')
        spsolve_triangular(A, b, unit_diagonal=True)