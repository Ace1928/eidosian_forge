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
def test_register_solve(self) -> None:
    Problem.register_solve('test', lambda self: 1)
    p = Problem(cp.Minimize(1))
    result = p.solve(method='test')
    self.assertEqual(result, 1)

    def test(self, a, b: int=2):
        return (a, b)
    Problem.register_solve('test', test)
    p = Problem(cp.Minimize(0))
    result = p.solve(1, b=3, method='test')
    self.assertEqual(result, (1, 3))
    result = p.solve(1, method='test')
    self.assertEqual(result, (1, 2))
    result = p.solve(1, method='test', b=4)
    self.assertEqual(result, (1, 4))