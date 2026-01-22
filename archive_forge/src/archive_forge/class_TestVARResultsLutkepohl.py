from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
class TestVARResultsLutkepohl:
    """
    Verify calculations using results from Lütkepohl's book
    """

    @classmethod
    def setup_class(cls):
        cls.p = 2
        sdata, dates = get_lutkepohl_data('e1')
        data = data_util.struct_to_ndarray(sdata)
        adj_data = np.diff(np.log(data), axis=0)
        cls.model = VAR(adj_data[:-16], dates=dates[1:-16], freq=f'B{QUARTER_END}-MAR')
        cls.res = cls.model.fit(maxlags=cls.p)
        cls.irf = cls.res.irf(10)
        cls.lut = E1_Results()

    def test_approx_mse(self):
        mse2 = np.array([[25.12, 0.58, 1.3], [0.58, 1.581, 0.586], [1.3, 0.586, 1.009]]) * 0.0001
        assert_almost_equal(mse2, self.res.forecast_cov(3)[1], DECIMAL_3)

    def test_irf_stderr(self):
        irf_stderr = self.irf.stderr(orth=False)
        for i in range(1, 1 + len(self.lut.irf_stderr)):
            assert_almost_equal(np.round(irf_stderr[i], 3), self.lut.irf_stderr[i - 1])

    def test_cum_irf_stderr(self):
        stderr = self.irf.cum_effect_stderr(orth=False)
        for i in range(1, 1 + len(self.lut.cum_irf_stderr)):
            assert_almost_equal(np.round(stderr[i], 3), self.lut.cum_irf_stderr[i - 1])

    def test_lr_effect_stderr(self):
        stderr = self.irf.lr_effect_stderr(orth=False)
        orth_stderr = self.irf.lr_effect_stderr(orth=True)
        assert_almost_equal(np.round(stderr, 3), self.lut.lr_stderr)