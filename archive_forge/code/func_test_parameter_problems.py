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
def test_parameter_problems(self) -> None:
    """Test problems with parameters.
        """
    p1 = Parameter()
    p2 = Parameter(3, nonpos=True)
    p3 = Parameter((4, 4), nonneg=True)
    p = Problem(cp.Maximize(p1 * self.a), [self.a + p1 <= p2, self.b <= p3 + p3 + 2])
    p1.value = 2
    p2.value = -numpy.ones((3,))
    p3.value = numpy.ones((4, 4))
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, -6)
    p1.value = None
    with self.assertRaises(ParameterError):
        p.solve(solver=cp.SCS, eps=1e-06)