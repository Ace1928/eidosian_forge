import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def _run_solout_test(self, integrator):
    ts = []
    ys = []
    t0 = 0.0
    tend = 20.0
    y0 = [0.0]

    def solout(t, y):
        ts.append(t)
        ys.append(y.copy())

    def rhs(t, y):
        return [1.0 / (t - 10.0 - 1j)]
    ig = complex_ode(rhs).set_integrator(integrator)
    ig.set_solout(solout)
    ig.set_initial_value(y0, t0)
    ret = ig.integrate(tend)
    assert_array_equal(ys[0], y0)
    assert_array_equal(ys[-1], ret)
    assert_equal(ts[0], t0)
    assert_equal(ts[-1], tend)