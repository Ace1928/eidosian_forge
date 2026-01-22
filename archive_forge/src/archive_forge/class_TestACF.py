from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
class TestACF(CheckCorrGram):
    """
    Test Autocorrelation Function
    """

    @classmethod
    def setup_class(cls):
        cls.acf = cls.results['acvar']
        cls.qstat = cls.results['Q1']
        cls.res1 = acf(cls.x, nlags=40, qstat=True, alpha=0.05, fft=False)
        cls.confint_res = cls.results[['acvar_lb', 'acvar_ub']].values

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:41], self.acf, DECIMAL_8)

    def test_confint(self):
        centered = self.res1[1] - self.res1[1].mean(1)[:, None]
        assert_almost_equal(centered[1:41], self.confint_res, DECIMAL_8)

    def test_qstat(self):
        assert_almost_equal(self.res1[2][:40], self.qstat, DECIMAL_3)