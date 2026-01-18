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
def test_pnorm_axis(self) -> None:
    """Test pnorm with axis != 0.
        """
    b = numpy.arange(2)
    X = cp.Variable(shape=(2, 10))
    expr = cp.pnorm(X, p=2, axis=1) - b
    con = [expr <= 0]
    obj = cp.Maximize(cp.sum(X))
    prob = cp.Problem(obj, con)
    prob.solve(solver=cp.ECOS)
    self.assertItemsAlmostEqual(expr.value, numpy.zeros(2))
    b = numpy.arange(10)
    X = cp.Variable(shape=(10, 2))
    expr = cp.pnorm(X, p=2, axis=1) - b
    con = [expr <= 0]
    obj = cp.Maximize(cp.sum(X))
    prob = cp.Problem(obj, con)
    prob.solve(solver=cp.ECOS)
    self.assertItemsAlmostEqual(expr.value, numpy.zeros(10))