import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
@pytest.mark.skip(reason='Failing in Windows CI - potentially memory leak')
def test_small_parameterized_lp(self) -> None:
    m = 200
    n = 200
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    c = cp.Parameter(n)
    A.value = np.random.randn(m, n)
    b.value = np.random.randn(m)
    c.value = np.random.rand(n)
    x = cp.Variable(n)
    cost = cp.matmul(c, x)
    constraints = [A @ x <= b]
    problem = cp.Problem(cp.Minimize(cost), constraints)

    def small_parameterized_lp():
        problem.get_problem_data(cp.SCS)
    benchmark(small_parameterized_lp, iters=1)
    benchmark(small_parameterized_lp, iters=1, name='small_parameterized_lp_second_time')