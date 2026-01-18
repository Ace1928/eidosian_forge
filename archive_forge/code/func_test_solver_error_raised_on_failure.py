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
def test_solver_error_raised_on_failure(self) -> None:
    """Tests that a SolverError is raised when a solver fails.
        """
    A = numpy.random.randn(40, 40)
    b = cp.matmul(A, numpy.random.randn(40))
    with self.assertRaises(SolverError):
        Problem(cp.Minimize(cp.sum_squares(cp.matmul(A, cp.Variable(40)) - b))).solve(solver=s.OSQP, max_iter=1)