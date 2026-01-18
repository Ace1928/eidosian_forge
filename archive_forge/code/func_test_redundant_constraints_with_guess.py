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
def test_redundant_constraints_with_guess(self):
    A, b, c, _, _ = magic_square(3)
    p = np.random.rand(*c.shape)
    with suppress_warnings() as sup:
        sup.filter(OptimizeWarning, 'A_eq does not appear...')
        sup.filter(RuntimeWarning, 'invalid value encountered')
        sup.filter(LinAlgWarning)
        res = linprog(c, A_eq=A, b_eq=b, method=self.method)
        res2 = linprog(c, A_eq=A, b_eq=b, method=self.method, x0=res.x)
        res3 = linprog(c + p, A_eq=A, b_eq=b, method=self.method, x0=res.x)
    _assert_success(res2, desired_fun=1.730550597)
    assert_equal(res2.nit, 0)
    _assert_success(res3)
    assert_(res3.nit < res.nit)