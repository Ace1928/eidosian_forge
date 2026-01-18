from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from platform import python_implementation
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import lgmres, gmres
def test_outer_v(self):
    outer_v = []
    x0, count_0 = do_solve(outer_k=6, outer_v=outer_v)
    assert_(len(outer_v) > 0)
    assert_(len(outer_v) <= 6)
    x1, count_1 = do_solve(outer_k=6, outer_v=outer_v, prepend_outer_v=True)
    assert_(count_1 == 2, count_1)
    assert_(count_1 < count_0 / 2)
    assert_(allclose(x1, x0, rtol=1e-14))
    outer_v = []
    x0, count_0 = do_solve(outer_k=6, outer_v=outer_v, store_outer_Av=False)
    assert_(array([v[1] is None for v in outer_v]).all())
    assert_(len(outer_v) > 0)
    assert_(len(outer_v) <= 6)
    x1, count_1 = do_solve(outer_k=6, outer_v=outer_v, prepend_outer_v=True)
    assert_(count_1 == 3, count_1)
    assert_(count_1 < count_0 / 2)
    assert_(allclose(x1, x0, rtol=1e-14))