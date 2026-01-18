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
def test_singleton_row_eq_1(self):
    c = [1, 1, 1, 2]
    A_eq = [[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]
    b_eq = [1, 2, 2, 4]
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_infeasible(res)
    if self.options.get('presolve', True):
        assert_equal(res.nit, 0)