import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_vode(self):
    for problem_cls in PROBLEMS:
        problem = problem_cls()
        if not problem.stiff:
            self._do_problem(problem, 'vode', 'adams')
        else:
            self._do_problem(problem, 'vode', 'bdf')