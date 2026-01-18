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
def test_special_index(self) -> None:
    """Test QP code path with special indexing.
        """
    x = cp.Variable((1, 3))
    y = cp.sum(x[:, 0:2], axis=1)
    cost = cp.QuadForm(y, np.diag([1]))
    prob = cp.Problem(cp.Minimize(cost))
    result1 = prob.solve(solver=cp.SCS)
    x = cp.Variable((1, 3))
    y = cp.sum(x[:, [0, 1]], axis=1)
    cost = cp.QuadForm(y, np.diag([1]))
    prob = cp.Problem(cp.Minimize(cost))
    result2 = prob.solve(solver=cp.SCS)
    self.assertAlmostEqual(result1, result2)