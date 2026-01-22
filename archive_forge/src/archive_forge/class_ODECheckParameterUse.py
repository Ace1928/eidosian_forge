import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
class ODECheckParameterUse:
    """Call an ode-class solver with several cases of parameter use."""
    solver_name = ''
    solver_uses_jac = False

    def _get_solver(self, f, jac):
        solver = ode(f, jac)
        if self.solver_uses_jac:
            solver.set_integrator(self.solver_name, atol=1e-09, rtol=1e-07, with_jacobian=self.solver_uses_jac)
        else:
            solver.set_integrator(self.solver_name, atol=1e-09, rtol=1e-07)
        return solver

    def _check_solver(self, solver):
        ic = [1.0, 0.0]
        solver.set_initial_value(ic, 0.0)
        solver.integrate(pi)
        assert_array_almost_equal(solver.y, [-1.0, 0.0])

    def test_no_params(self):
        solver = self._get_solver(f, jac)
        self._check_solver(solver)

    def test_one_scalar_param(self):
        solver = self._get_solver(f1, jac1)
        omega = 1.0
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)

    def test_two_scalar_params(self):
        solver = self._get_solver(f2, jac2)
        omega1 = 1.0
        omega2 = 1.0
        solver.set_f_params(omega1, omega2)
        if self.solver_uses_jac:
            solver.set_jac_params(omega1, omega2)
        self._check_solver(solver)

    def test_vector_param(self):
        solver = self._get_solver(fv, jacv)
        omega = [1.0, 1.0]
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)

    def test_warns_on_failure(self):
        solver = self._get_solver(f, jac)
        solver.set_integrator(self.solver_name, nsteps=1)
        ic = [1.0, 0.0]
        solver.set_initial_value(ic, 0.0)
        assert_warns(UserWarning, solver.integrate, pi)