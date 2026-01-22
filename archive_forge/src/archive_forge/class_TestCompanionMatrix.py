import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
class TestCompanionMatrix:
    cases = [(2, np.array([[0, 1], [0, 0]])), ([1, -1, -2], np.array([[1, 1], [2, 0]])), ([1, -1, -2, -3], np.array([[1, 1, 0], [2, 0, 1], [3, 0, 0]])), ([1, -np.array([[1, 2], [3, 4]]), -np.array([[5, 6], [7, 8]])], np.array([[1, 2, 5, 6], [3, 4, 7, 8], [1, 0, 0, 0], [0, 1, 0, 0]]).T), (np.int64(2), np.array([[0, 1], [0, 0]]))]

    def test_cases(self):
        for polynomial, result in self.cases:
            assert_equal(tools.companion_matrix(polynomial), result)