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
def test_psd_constraints(self) -> None:
    """Test positive definite constraints.
        """
    C = Variable((3, 3))
    obj = cp.Maximize(C[0, 2])
    constraints = [cp.diag(C) == 1, C[0, 1] == 0.6, C[1, 2] == -0.3, C == C.T, C >> 0]
    prob = Problem(obj, constraints)
    result = prob.solve(solver=cp.SCS)
    self.assertAlmostEqual(result, 0.583151, places=2)
    C = Variable((2, 2))
    obj = cp.Maximize(C[0, 1])
    constraints = [C == 1, C >> [[2, 0], [0, 2]]]
    prob = Problem(obj, constraints)
    result = prob.solve(solver=cp.SCS)
    self.assertEqual(prob.status, s.INFEASIBLE)
    C = Variable((2, 2), symmetric=True)
    obj = cp.Minimize(C[0, 0])
    constraints = [C << [[2, 0], [0, 2]]]
    prob = Problem(obj, constraints)
    result = prob.solve(solver=cp.SCS)
    self.assertEqual(prob.status, s.UNBOUNDED)