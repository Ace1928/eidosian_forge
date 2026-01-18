import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def test_positional_deprecation(solver):
    rng = np.random.default_rng(1685363802304750)
    n = 10
    A = rng.random(size=[n, n])
    A = A @ A.T
    b = rng.random(n)
    x0 = rng.random(n)
    with pytest.deprecated_call(match='use keyword arguments.*|argument `tol` is deprecated.*'):
        solver(A, b, x0, 1e-05)