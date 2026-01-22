import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
class TestIsInvertible:
    cases = [([1, -0.5], True), ([1, 1 - 1e-09], True), ([1, 1], False), ([1, 0.9, 0.1], True), (np.array([1, 0.9, 0.1]), True), (pd.Series([1, 0.9, 0.1]), True)]

    def test_cases(self):
        for polynomial, invertible in self.cases:
            assert_equal(tools.is_invertible(polynomial), invertible)