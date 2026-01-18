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
def test_optimize_result(self):
    c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(0)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
    assert_(res.success)
    assert_(res.nit)
    assert_(not res.status)
    if 'highs' not in self.method:
        assert_(res.message == 'Optimization terminated successfully.')
    assert_allclose(c @ res.x, res.fun)
    assert_allclose(b_eq - A_eq @ res.x, res.con, atol=1e-11)
    assert_allclose(b_ub - A_ub @ res.x, res.slack, atol=1e-11)
    for key in ['eqlin', 'ineqlin', 'lower', 'upper']:
        if key in res.keys():
            assert isinstance(res[key]['marginals'], np.ndarray)
            assert isinstance(res[key]['residual'], np.ndarray)