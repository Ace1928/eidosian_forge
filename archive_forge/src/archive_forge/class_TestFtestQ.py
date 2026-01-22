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
class TestFtestQ:
    """
    A joint hypothesis test that Rb = q.  Coefficient tests are essentially
    made up.  Test values taken from Stata.
    """

    @classmethod
    def setup_class(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        res1 = OLS(data.endog, data.exog).fit()
        R = np.array([[0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0]])
        q = np.array([0, 0, 0, 1, 0])
        cls.Ftest1 = res1.f_test((R, q))

    def test_fvalue(self):
        assert_almost_equal(self.Ftest1.fvalue, 70.115557, 5)

    def test_pvalue(self):
        assert_almost_equal(self.Ftest1.pvalue, 6.229e-07, 10)

    def test_df_denom(self):
        assert_equal(self.Ftest1.df_denom, 9)

    def test_df_num(self):
        assert_equal(self.Ftest1.df_num, 5)