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
def test_sparse_constraints(self):

    def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    np.random.seed(0)
    m = 100
    n = 150
    A_eq = scipy.sparse.rand(m, n, 0.5)
    x_valid = np.random.randn(n)
    c = np.random.randn(n)
    ub = x_valid + np.random.rand(n)
    lb = x_valid - np.random.rand(n)
    bounds = np.column_stack((lb, ub))
    b_eq = A_eq * x_valid
    if self.method in {'simplex', 'revised simplex'}:
        with assert_raises(ValueError, match=f"Method '{self.method}' does not support sparse constraint matrices."):
            linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
    else:
        options = {**self.options}
        if self.method in {'interior-point'}:
            options['sparse'] = True
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=options)
        assert res.success