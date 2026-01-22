import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
class TestOde(TestODEClass):
    ode_class = ode

    def test_vode(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if not problem.stiff:
                self._do_problem(problem, 'vode', 'adams')
            self._do_problem(problem, 'vode', 'bdf')

    def test_zvode(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if not problem.stiff:
                self._do_problem(problem, 'zvode', 'adams')
            self._do_problem(problem, 'zvode', 'bdf')

    def test_lsoda(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            self._do_problem(problem, 'lsoda')

    def test_dopri5(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dopri5')

    def test_dop853(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dop853')

    def test_concurrent_fail(self):
        for sol in ('vode', 'zvode', 'lsoda'):

            def f(t, y):
                return 1.0
            r = ode(f).set_integrator(sol)
            r.set_initial_value(0, 0)
            r2 = ode(f).set_integrator(sol)
            r2.set_initial_value(0, 0)
            r.integrate(r.t + 0.1)
            r2.integrate(r2.t + 0.1)
            assert_raises(RuntimeError, r.integrate, r.t + 0.1)

    def test_concurrent_ok(self):

        def f(t, y):
            return 1.0
        for k in range(3):
            for sol in ('vode', 'zvode', 'lsoda', 'dopri5', 'dop853'):
                r = ode(f).set_integrator(sol)
                r.set_initial_value(0, 0)
                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r2.integrate(r2.t + 0.1)
                assert_allclose(r.y, 0.1)
                assert_allclose(r2.y, 0.2)
            for sol in ('dopri5', 'dop853'):
                r = ode(f).set_integrator(sol)
                r.set_initial_value(0, 0)
                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)
                r.integrate(r.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                assert_allclose(r.y, 0.3)
                assert_allclose(r2.y, 0.2)