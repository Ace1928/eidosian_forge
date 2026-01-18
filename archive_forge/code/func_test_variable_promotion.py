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
def test_variable_promotion(self) -> None:
    p = Problem(cp.Minimize(self.a), [self.x <= self.a, self.x == [1, 2]])
    result = p.solve(solver=cp.ECOS)
    self.assertAlmostEqual(result, 2)
    self.assertAlmostEqual(self.a.value, 2)
    p = Problem(cp.Minimize(self.a), [self.A <= self.a, self.A == [[1, 2], [3, 4]]])
    result = p.solve(solver=cp.ECOS)
    self.assertAlmostEqual(result, 4)
    self.assertAlmostEqual(self.a.value, 4)
    p = Problem(cp.Minimize([[1], [1]] @ (self.x + self.a + 1)), [self.a + self.x >= [1, 2]])
    result = p.solve(solver=cp.ECOS)
    self.assertAlmostEqual(result, 5)