import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
@pytest.mark.skip(reason='Failing in Windows CI - potentially memory leak')
def test_small_parameterized_cone_matrix_stuffing(self) -> None:
    m = 200
    n = 200
    A = cp.Parameter((m, n))
    C = cp.Parameter(m // 2)
    b = cp.Parameter(m)
    A.value = np.random.randn(m, n)
    C.value = np.random.rand(m // 2)
    b.value = np.random.randn(m)
    x = cp.Variable(n)
    cost = cp.sum(A @ x)
    constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
    constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])
    problem = cp.Problem(cp.Minimize(cost), constraints)

    def small_parameterized_cone_matrix_stuffing():
        ConeMatrixStuffing().apply(problem)
    benchmark(small_parameterized_cone_matrix_stuffing, iters=1)