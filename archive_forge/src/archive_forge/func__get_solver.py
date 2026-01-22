import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def _get_solver(self, f, jac):
    solver = ode(f, jac)
    if self.solver_uses_jac:
        solver.set_integrator(self.solver_name, atol=1e-09, rtol=1e-07, with_jacobian=self.solver_uses_jac)
    else:
        solver.set_integrator(self.solver_name, atol=1e-09, rtol=1e-07)
    return solver