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
def test_solver_stats(self) -> None:
    """Test the solver_stats method.
        """
    prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
    prob.solve(solver=s.ECOS)
    stats = prob.solver_stats
    self.assertGreater(stats.solve_time, 0)
    self.assertGreater(stats.setup_time, 0)
    self.assertGreater(stats.num_iters, 0)
    self.assertIn('info', stats.extra_stats)
    prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
    prob.solve(solver=s.SCS)
    stats = prob.solver_stats
    self.assertGreater(stats.solve_time, 0)
    self.assertGreater(stats.setup_time, 0)
    self.assertGreater(stats.num_iters, 0)
    self.assertIn('info', stats.extra_stats)
    prob = Problem(cp.Minimize(cp.sum(self.x)), [self.x == 0])
    prob.solve(solver=s.OSQP)
    stats = prob.solver_stats
    self.assertGreater(stats.solve_time, 0)
    self.assertGreater(stats.num_iters, 0)
    self.assertTrue(hasattr(stats.extra_stats, 'info'))
    self.assertTrue(str(stats).startswith('SolverStats(solver_name='))