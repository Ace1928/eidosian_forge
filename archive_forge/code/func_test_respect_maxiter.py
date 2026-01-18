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
@pytest.mark.parametrize('method', ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'])
def test_respect_maxiter(self, method):
    MAXITER = 4
    x0 = np.zeros(10)
    sf = ScalarFunction(optimize.rosen, x0, (), optimize.rosen_der, optimize.rosen_hess, None, None)
    kwargs = {'method': method, 'options': dict(maxiter=MAXITER)}
    if method in ('Newton-CG',):
        kwargs['jac'] = sf.grad
    elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg', 'trust-constr'):
        kwargs['jac'] = sf.grad
        kwargs['hess'] = sf.hess
    sol = optimize.minimize(sf.fun, x0, **kwargs)
    assert sol.nit == MAXITER
    assert sol.nfev >= sf.nfev
    if hasattr(sol, 'njev'):
        assert sol.njev >= sf.ngev
    if method == 'SLSQP':
        assert sol.status == 9