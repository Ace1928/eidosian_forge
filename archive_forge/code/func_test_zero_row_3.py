import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def test_zero_row_3(self):
    m, n = (2, 4)
    c = np.random.rand(n)
    A_eq = np.random.rand(m, n)
    A_eq[0, :] = 0
    b_eq = np.random.rand(m)
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_infeasible(res)
    if self.options.get('presolve', True):
        assert_equal(res.nit, 0)