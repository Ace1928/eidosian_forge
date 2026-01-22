from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
class TestYuleWalker:

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.sunspots import load
        data = load()
        cls.rho, cls.sigma = yule_walker(data.endog, order=4, method='mle')
        cls.R_params = [1.2831003105694765, -0.45240924374091945, -0.20770298557575195, 0.04794364808954234]

    def test_params(self):
        assert_almost_equal(self.rho, self.R_params, DECIMAL_4)