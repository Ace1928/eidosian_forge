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
def test_size_metrics(self) -> None:
    """Test the size_metrics method.
        """
    p1 = Parameter()
    p2 = Parameter(3, nonpos=True)
    p3 = Parameter((4, 4), nonneg=True)
    c1 = numpy.random.randn(2, 1)
    c2 = numpy.random.randn(1, 2)
    constants = [2, c2.dot(c1)]
    p = Problem(cp.Minimize(p1), [self.a + p1 <= p2, self.b <= p3 + p3 + constants[0], self.c == constants[1]])
    n_variables = p.size_metrics.num_scalar_variables
    ref = self.a.size + self.b.size + self.c.size
    self.assertEqual(n_variables, ref)
    n_data = p.size_metrics.num_scalar_data
    ref = numpy.prod(p1.size) + numpy.prod(p2.size) + numpy.prod(p3.size) + len(constants)
    self.assertEqual(n_data, ref)
    n_eq_constr = p.size_metrics.num_scalar_eq_constr
    ref = c2.dot(c1).size
    self.assertEqual(n_eq_constr, ref)
    n_leq_constr = p.size_metrics.num_scalar_leq_constr
    ref = numpy.prod(p3.size) + numpy.prod(p2.size)
    self.assertEqual(n_leq_constr, ref)
    max_data_dim = p.size_metrics.max_data_dimension
    ref = max(p3.shape)
    self.assertEqual(max_data_dim, ref)