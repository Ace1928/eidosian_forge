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
def test_t_eval():
    rtol = 0.001
    atol = 1e-06
    y0 = [1 / 3, 2 / 9]
    for t_span in ([5, 9], [5, 1]):
        t_eval = np.linspace(t_span[0], t_span[1], 10)
        res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol, t_eval=t_eval)
        assert_equal(res.t, t_eval)
        assert_(res.t_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)
        y_true = sol_rational(res.t)
        e = compute_error(res.y, y_true, rtol, atol)
        assert_(np.all(e < 5))
    t_eval = [5, 5.01, 7, 8, 8.01, 9]
    res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol, t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))
    t_eval = [5, 4.99, 3, 1.5, 1.1, 1.01, 1]
    res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol, t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)
    t_eval = [5.01, 7, 8, 8.01]
    res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol, t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))
    t_eval = [4.99, 3, 1.5, 1.1, 1.01]
    res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol, t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)
    t_eval = [4, 6]
    assert_raises(ValueError, solve_ivp, fun_rational, [5, 9], y0, rtol=rtol, atol=atol, t_eval=t_eval)