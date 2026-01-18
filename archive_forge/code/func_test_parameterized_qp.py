import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def test_parameterized_qp(self) -> None:
    self.skipTest('This benchmark takes too long.')
    'Test speed of first solve with QP codepath and SOCP codepath.\n        '
    m = 150
    n = 100
    np.random.seed(1)
    A = cp.Parameter((m, n))
    b = cp.Parameter((m,))
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)
    start = time.time()
    A.value = np.random.randn(m, n)
    b.value = np.random.randn(m)
    prob.solve(solver=cp.ECOS)
    end = time.time()
    print('Conic canonicalization')
    print('(ECOS) solver time: ', prob.solver_stats.solve_time)
    print('cvxpy time: ', end - start - prob.solver_stats.solve_time)
    np.random.seed(1)
    A = cp.Parameter((m, n))
    b = cp.Parameter((m,))
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)
    start = time.time()
    A.value = np.random.randn(m, n)
    b.value = np.random.randn(m)
    prob.solve(solver=cp.OSQP)
    end = time.time()
    print('Quadratic canonicalization')
    print('(OSQP) solver time: ', prob.solver_stats.solve_time)
    print('cvxpy time: ', end - start - prob.solver_stats.solve_time)