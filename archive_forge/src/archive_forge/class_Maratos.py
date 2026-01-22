import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
class Maratos:
    """Problem 15.4 from Nocedal and Wright

    The following optimization problem:
        minimize 2*(x[0]**2 + x[1]**2 - 1) - x[0]
        Subject to: x[0]**2 + x[1]**2 - 1 = 0
    """

    def __init__(self, degrees=60, constr_jac=None, constr_hess=None):
        rads = degrees / 180 * np.pi
        self.x0 = [np.cos(rads), np.sin(rads)]
        self.x_opt = np.array([1.0, 0.0])
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.bounds = None

    def fun(self, x):
        return 2 * (x[0] ** 2 + x[1] ** 2 - 1) - x[0]

    def grad(self, x):
        return np.array([4 * x[0] - 1, 4 * x[1]])

    def hess(self, x):
        return 4 * np.eye(2)

    @property
    def constr(self):

        def fun(x):
            return x[0] ** 2 + x[1] ** 2
        if self.constr_jac is None:

            def jac(x):
                return [[2 * x[0], 2 * x[1]]]
        else:
            jac = self.constr_jac
        if self.constr_hess is None:

            def hess(x, v):
                return 2 * v[0] * np.eye(2)
        else:
            hess = self.constr_hess
        return NonlinearConstraint(fun, 1, 1, jac, hess)