import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
@staticmethod
def vec_pow_nd() -> STH.SolverTestHelper:
    n_buyer = 4
    n_items = 6
    z = cp.Variable(shape=(2,))
    np.random.seed(1)
    V = np.random.rand(n_buyer, n_items)
    X = cp.Variable(shape=(n_buyer, n_items), nonneg=True)
    u = cp.sum(cp.multiply(V, X), axis=1)
    alpha1 = np.array([0.02999541, 0.24340343, 0.03687151, 0.68972966])
    alpha2 = np.array([0.24041855, 0.1745123, 0.10012628, 0.48494287])
    cons = [cp.PowConeND(cp.vstack([u, u]), z, np.vstack([alpha1, alpha2]), axis=1), X >= 0, cp.sum(X, axis=0) <= 1]
    obj = cp.Maximize(z[0] + z[1])
    prob = cp.Problem(obj, cons)
    prob.solve(solver='SCS')
    obj_pair = (obj, 2.7003780870341516)
    cons_pairs = [(con, None) for con in cons]
    var_pairs = [(z, None), (X, None), (u, None)]
    return STH.SolverTestHelper(obj_pair, var_pairs, cons_pairs)