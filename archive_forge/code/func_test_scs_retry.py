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
def test_scs_retry(self) -> None:
    """Test that SCS retry doesn't trigger a crash.
        """
    n_sec = 20
    np.random.seed(315)
    mu = np.random.random(n_sec)
    random_mat = np.random.rand(n_sec, n_sec)
    C = np.dot(random_mat, random_mat.transpose())
    x = cp.Variable(n_sec)
    prob = cp.Problem(cp.Minimize(cp.QuadForm(x, C)), [cp.sum(x) == 1, 0 <= x, x <= 1, x @ mu >= np.max(mu) - 1e-06])
    prob.solve(cp.SCS)
    assert prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}