import builtins
import pickle
import sys
import warnings
from fractions import Fraction
from io import StringIO
import ecos
import numpy
import numpy as np
import scipy.sparse as sp
import scs
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import PSD, ExpCone, NonNeg, Zero
from cvxpy.error import DCPError, ParameterError, SolverError
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.conic_solvers import ecos_conif, scs_conif
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import (
from cvxpy.reductions.solvers.solving_chain import ECOS_DEPRECATION_MSG
from cvxpy.tests.base_test import BaseTest
def test_rmul_scalar_mats(self) -> None:
    """Test that rmul works with 1x1 matrices.
        """
    x = [[4144.30127531]]
    y = [[7202.52114311]]
    z = cp.Variable(shape=(1, 1))
    objective = cp.Minimize(cp.quad_form(z, x) - 2 * z.T @ y)
    prob = cp.Problem(objective)
    prob.solve(cp.OSQP, verbose=True)
    result1 = prob.value
    x = 4144.30127531
    y = 7202.52114311
    z = cp.Variable()
    objective = cp.Minimize(x * z ** 2 - 2 * z * y)
    prob = cp.Problem(objective)
    prob.solve(cp.OSQP, verbose=True)
    self.assertAlmostEqual(prob.value, result1)