import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def test_nonlin_bc():
    x = np.linspace(0, 0.1, 5)
    x_test = x
    y = np.zeros([2, x.size])
    sol = solve_bvp(nonlin_bc_fun, nonlin_bc_bc, x, y)
    assert_equal(sol.status, 0)
    assert_(sol.success)
    assert_(sol.x.size < 8)
    sol_test = sol.sol(x_test)
    assert_allclose(sol_test[0], nonlin_bc_sol(x_test), rtol=1e-05, atol=1e-05)
    f_test = nonlin_bc_fun(x_test, sol_test)
    r = sol.sol(x_test, 1) - f_test
    rel_res = r / (1 + np.abs(f_test))
    norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5
    assert_(np.all(norm_res < 0.001))
    assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)