from itertools import product
from numpy.testing import (assert_, assert_allclose, assert_array_less,
import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._numdiff import group_columns
from scipy.integrate import solve_ivp, RK23, RK45, DOP853, Radau, BDF, LSODA
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.common import num_jac
from scipy.integrate._ivp.base import ConstantDenseOutput
from scipy.sparse import coo_matrix, csc_matrix
def test_t_eval_dense_output():
    rtol = 0.001
    atol = 1e-06
    y0 = [1 / 3, 2 / 9]
    t_span = [5, 9]
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol, t_eval=t_eval)
    res_d = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=True)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)
    assert_equal(res.t, res_d.t)
    assert_equal(res.y, res_d.y)
    assert_(res_d.t_events is None)
    assert_(res_d.success)
    assert_equal(res_d.status, 0)
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))