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
class TestNxNx(TestDataDimensions):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.mod2 = OLS(cls.endog_n_, cls.exog_n_)
        cls.mod2.df_model += 1
        cls.res2 = cls.mod2.fit()