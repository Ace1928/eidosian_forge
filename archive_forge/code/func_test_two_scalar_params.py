import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_two_scalar_params(self):
    solver = self._get_solver(f2, jac2)
    omega1 = 1.0
    omega2 = 1.0
    solver.set_f_params(omega1, omega2)
    if self.solver_uses_jac:
        solver.set_jac_params(omega1, omega2)
    self._check_solver(solver)