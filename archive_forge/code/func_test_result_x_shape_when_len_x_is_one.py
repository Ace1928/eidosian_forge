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
def test_result_x_shape_when_len_x_is_one():

    def fun(x):
        return x * x

    def jac(x):
        return 2.0 * x

    def hess(x):
        return np.array([[2.0]])
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
    for method in methods:
        res = optimize.minimize(fun, np.array([0.1]), method=method)
        assert res.x.shape == (1,)
    methods = ['trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov', 'Newton-CG']
    for method in methods:
        res = optimize.minimize(fun, np.array([0.1]), method=method, jac=jac, hess=hess)
        assert res.x.shape == (1,)