import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
class TestComplexOde(TestODEClass):
    ode_class = complex_ode

    def test_vode(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if not problem.stiff:
                self._do_problem(problem, 'vode', 'adams')
            else:
                self._do_problem(problem, 'vode', 'bdf')

    def test_lsoda(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            self._do_problem(problem, 'lsoda')

    def test_dopri5(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dopri5')

    def test_dop853(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dop853')