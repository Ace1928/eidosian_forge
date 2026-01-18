import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_quad_obj_with_power(self) -> None:
    """Test a mixed quadratic/power objective.
        """
    import scs
    if Version(scs.__version__) >= Version('3.0.0'):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x ** 1.6 + x ** 2), [x >= 1])
        prob.solve(solver=cp.SCS, use_quad_obj=True)
        self.assertAlmostEqual(prob.value, 2)
        self.assertAlmostEqual(x.value, 1)
        data = prob.get_problem_data(solver=cp.SCS, solver_opts={'use_quad_obj': True})
        assert 'P' in data[0]
        assert data[0]['dims'].soc