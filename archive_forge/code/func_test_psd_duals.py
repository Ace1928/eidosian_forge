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
def test_psd_duals(self) -> None:
    """Test the duals of PSD constraints.
        """
    if s.CVXOPT in INSTALLED_SOLVERS:
        C = Variable((2, 2), symmetric=True, name='C')
        obj = cp.Maximize(C[0, 0])
        constraints = [C << [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.CVXOPT)
        self.assertAlmostEqual(result, 2)
        psd_constr_dual = constraints[0].dual_value.copy()
        C = Variable((2, 2), symmetric=True, name='C')
        X = Variable((2, 2), PSD=True)
        obj = cp.Maximize(C[0, 0])
        constraints = [X == [[2, 0], [0, 2]] - C]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.CVXOPT)
        new_constr_dual = (constraints[0].dual_value + constraints[0].dual_value.T) / 2
        self.assertItemsAlmostEqual(new_constr_dual, psd_constr_dual)
    C = Variable((2, 2), symmetric=True)
    obj = cp.Maximize(C[0, 0])
    constraints = [C << [[2, 0], [0, 2]]]
    prob = Problem(obj, constraints)
    result = prob.solve(solver=s.SCS)
    self.assertAlmostEqual(result, 2, places=4)
    psd_constr_dual = constraints[0].dual_value
    C = Variable((2, 2), symmetric=True)
    X = Variable((2, 2), PSD=True)
    obj = cp.Maximize(C[0, 0])
    constraints = [X == [[2, 0], [0, 2]] - C]
    prob = Problem(obj, constraints)
    result = prob.solve(solver=s.SCS)
    self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual)
    C = Variable((2, 2), symmetric=True)
    obj = cp.Maximize(C[0, 1] + C[1, 0])
    constraints = [C << [[2, 0], [0, 2]], C >= 0]
    prob = Problem(obj, constraints)
    result = prob.solve(solver=s.SCS)
    self.assertAlmostEqual(result, 4, places=3)
    psd_constr_dual = constraints[0].dual_value
    C = Variable((2, 2), symmetric=True)
    X = Variable((2, 2), PSD=True)
    obj = cp.Maximize(C[0, 1] + C[1, 0])
    constraints = [X == [[2, 0], [0, 2]] - C, C >= 0]
    prob = Problem(obj, constraints)
    result = prob.solve(solver=s.SCS)
    self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual, places=3)