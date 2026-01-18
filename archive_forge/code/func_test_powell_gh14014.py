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
@pytest.mark.xfail(reason='This part of test_powell fails on some platforms, but the solution returned by powell is still valid.')
def test_powell_gh14014(self):
    if self.use_wrapper:
        opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
        res = optimize.minimize(self.func, self.startparams, args=(), method='Powell', options=opts)
        params, fopt, direc, numiter, func_calls, warnflag = (res['x'], res['fun'], res['direc'], res['nit'], res['nfev'], res['status'])
    else:
        retval = optimize.fmin_powell(self.func, self.startparams, args=(), maxiter=self.maxiter, full_output=True, disp=self.disp, retall=False)
        params, fopt, direc, numiter, func_calls, warnflag = retval
    assert_allclose(self.trace[34:39], [[0.72949016, -0.44156936, 0.47100962], [0.72949016, -0.44156936, 0.48052496], [1.45898031, -0.88313872, 0.95153458], [0.72949016, -0.44156936, 0.47576729], [1.72949016, -0.44156936, 0.47576729]], atol=1e-14, rtol=1e-07)