import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_power_portfolio(self) -> None:
    """Test the portfolio problem in issue #2042"""
    T, N = (200, 10)
    rs = np.random.RandomState(123)
    mean = np.zeros(N) + 1 / 1000
    cov = rs.rand(N, N) * 1.5 - 0.5
    cov = cov @ cov.T / 1000 + np.diag(rs.rand(N) * 0.7 + 0.3) / 1000
    Y = st.multivariate_normal.rvs(mean=mean, cov=cov, size=T, random_state=rs)
    w = cp.Variable((N, 1))
    t = cp.Variable((1, 1))
    z = cp.Variable((1, 1))
    omega = cp.Variable((T, 1))
    psi = cp.Variable((T, 1))
    nu = cp.Variable((T, 1))
    epsilon = cp.Variable((T, 1))
    k = cp.Variable((1, 1))
    b = np.ones((1, N)) / N
    X = Y @ w
    h = 0.2
    ones = np.ones((T, 1))
    constraints = [cp.constraints.power.PowCone3D(z * (1 + h) / (2 * h) * ones, psi * (1 + h) / h, epsilon, 1 / (1 + h)), cp.constraints.power.PowCone3D(omega / (1 - h), nu / h, -z / (2 * h) * ones, 1 - h), -X - t + epsilon + omega <= 0, w >= 0, z >= 0]
    obj = t + z + cp.sum(psi + nu)
    constraints += [cp.sum(w) == k, k >= 0, b @ cp.log(w) >= 1]
    objective = cp.Minimize(obj)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)
    assert prob.status is cp.OPTIMAL