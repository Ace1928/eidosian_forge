import unittest
import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
def test_scalar_product(self) -> None:
    """Test scalar product.
        """
    p = np.ones((4,))
    v = cp.Variable((4,))
    p = np.ones((4,))
    obj = cp.Minimize(cp.scalar_product(v, p))
    prob = cp.Problem(obj, [v >= 1])
    prob.solve(solver=cp.SCS)
    assert np.allclose(v.value, p)
    p = cp.Parameter((4,))
    v = cp.Variable((4,))
    p.value = np.ones((4,))
    obj = cp.Minimize(cp.scalar_product(v, p))
    prob = cp.Problem(obj, [v >= 1])
    prob.solve(solver=cp.SCS)
    assert np.allclose(v.value, p.value)