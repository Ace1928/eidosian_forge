import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_odeint_bad_shapes():

    def badrhs(x, t):
        return [1, -1]

    def sys1(x, t):
        return -100 * x

    def badjac(x, t):
        return [[0, 0, 0]]
    bad_y0 = [[0, 0], [0, 0]]
    assert_raises(ValueError, odeint, sys1, bad_y0, [0, 1])
    bad_t = [[0, 1], [2, 3]]
    assert_raises(ValueError, odeint, sys1, [10.0], bad_t)
    assert_raises(RuntimeError, odeint, badrhs, 10, [0, 1])
    assert_raises(RuntimeError, odeint, sys1, [10, 10], [0, 1], Dfun=badjac)