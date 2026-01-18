import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_odeint_errors():

    def sys1d(x, t):
        return -100 * x

    def bad1(x, t):
        return 1.0 / 0

    def bad2(x, t):
        return 'foo'

    def bad_jac1(x, t):
        return 1.0 / 0

    def bad_jac2(x, t):
        return [['foo']]

    def sys2d(x, t):
        return [-100 * x[0], -0.1 * x[1]]

    def sys2d_bad_jac(x, t):
        return [[1.0 / 0, 0], [0, -0.1]]
    assert_raises(ZeroDivisionError, odeint, bad1, 1.0, [0, 1])
    assert_raises(ValueError, odeint, bad2, 1.0, [0, 1])
    assert_raises(ZeroDivisionError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac1)
    assert_raises(ValueError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac2)
    assert_raises(ZeroDivisionError, odeint, sys2d, [1.0, 1.0], [0, 1], Dfun=sys2d_bad_jac)