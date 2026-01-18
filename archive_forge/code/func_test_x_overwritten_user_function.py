import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
def test_x_overwritten_user_function():

    def fquad(x):
        a = np.arange(np.size(x))
        x -= a
        x *= x
        return np.sum(x)

    def fquad_jac(x):
        a = np.arange(np.size(x))
        x *= 2
        x -= 2 * a
        return x

    def fquad_hess(x):
        return np.eye(np.size(x)) * 2.0
    meth_jac = ['newton-cg', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov', 'trust-constr']
    meth_hess = ['dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov', 'trust-constr']
    x0 = np.ones(5) * 1.5
    for meth in MINIMIZE_METHODS:
        jac = None
        hess = None
        if meth in meth_jac:
            jac = fquad_jac
        if meth in meth_hess:
            hess = fquad_hess
        res = optimize.minimize(fquad, x0, method=meth, jac=jac, hess=hess)
        assert_allclose(res.x, np.arange(np.size(x0)), atol=0.0002)