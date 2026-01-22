import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
class ExScalarFunction:

    def __init__(self):
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0

    def fun(self, x):
        self.nfev += 1
        return 2 * (x[0] ** 2 + x[1] ** 2 - 1) - x[0]

    def grad(self, x):
        self.ngev += 1
        return np.array([4 * x[0] - 1, 4 * x[1]])

    def hess(self, x):
        self.nhev += 1
        return 4 * np.eye(2)