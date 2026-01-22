import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
class TestConstrainStationaryUnivariate:
    cases = [(np.array([2.0]), -2.0 / (1 + 2.0 ** 2) ** 0.5)]

    def test_cases(self):
        for unconstrained, constrained in self.cases:
            result = tools.constrain_stationary_univariate(unconstrained)
            assert_equal(result, constrained)