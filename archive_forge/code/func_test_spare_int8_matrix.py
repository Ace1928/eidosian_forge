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
def test_spare_int8_matrix(self) -> None:
    """Test problem with sparse int8 matrix.
           issue #809.
        """
    a = Variable(shape=(3, 1))
    q = np.array([1.88922129, 0.06938685, 0.91948919])
    P = np.array([[280.64, -49.84, -80.0], [-49.84, 196.04, 139.0], [-80.0, 139.0, 106.0]])
    D_dense = np.array([[-1, 1, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0], [0, 0, 0, -1, 1, 0]], dtype=np.int8)
    D_sparse = sp.coo_matrix(D_dense)

    def make_problem(D):
        obj = cp.Minimize(0.5 * cp.quad_form(a, P) - a.T @ q)
        assert obj.is_dcp()
        alpha = cp.Parameter(nonneg=True, value=2)
        constraints = [a >= 0.0, -alpha <= D.T @ a, D.T @ a <= alpha]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.settings.ECOS)
        assert prob.status == 'optimal'
        return prob
    expected_coef = np.array([[-0.011728003147, 0.011728002895, 2.52e-10, -0.017524801335, 0.017524801335, 0.0]])
    make_problem(D_dense)
    coef_dense = a.value.T.dot(D_dense)
    np.testing.assert_almost_equal(expected_coef, coef_dense)
    make_problem(D_sparse)
    coef_sparse = a.value.T @ D_sparse
    np.testing.assert_almost_equal(expected_coef, coef_sparse)