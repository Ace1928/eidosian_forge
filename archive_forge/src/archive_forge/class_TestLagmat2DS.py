from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
class TestLagmat2DS:

    @classmethod
    def setup_class(cls):
        data = macrodata.load_pandas()
        cls.macro_df = data.data[['year', 'quarter', 'realgdp', 'cpi']]
        np.random.seed(12345)
        cls.random_data = np.random.randn(100)
        index = [str(int(yr)) + '-Q' + str(int(qu)) for yr, qu in zip(cls.macro_df.year, cls.macro_df.quarter)]
        cls.macro_df.index = index
        cls.series = cls.macro_df.cpi

    @staticmethod
    def _prepare_expected(data, lags, trim='front'):
        t, k = data.shape
        expected = np.zeros((t + lags, (lags + 1) * k))
        for col in range(k):
            for i in range(lags + 1):
                if i < lags:
                    expected[i:-lags + i, (lags + 1) * col + i] = data[:, col]
                else:
                    expected[i:, (lags + 1) * col + i] = data[:, col]
        if trim == 'front':
            expected = expected[:-lags]
        return expected

    def test_lagmat2ds_numpy(self):
        data = self.macro_df
        npdata = data.values
        lagmat = stattools.lagmat2ds(npdata, 2)
        expected = self._prepare_expected(npdata, 2)
        assert_array_equal(lagmat, expected)
        lagmat = stattools.lagmat2ds(npdata[:, :2], 3)
        expected = self._prepare_expected(npdata[:, :2], 3)
        assert_array_equal(lagmat, expected)
        npdata = self.series.values
        lagmat = stattools.lagmat2ds(npdata, 5)
        expected = self._prepare_expected(npdata[:, None], 5)
        assert_array_equal(lagmat, expected)

    def test_lagmat2ds_pandas(self):
        data = self.macro_df
        lagmat = stattools.lagmat2ds(data, 2)
        expected = self._prepare_expected(data.values, 2)
        assert_array_equal(lagmat, expected)
        lagmat = stattools.lagmat2ds(data.iloc[:, :2], 3, trim='both')
        expected = self._prepare_expected(data.values[:, :2], 3)
        expected = expected[3:]
        assert_array_equal(lagmat, expected)
        data = self.series
        lagmat = stattools.lagmat2ds(data, 5)
        expected = self._prepare_expected(data.values[:, None], 5)
        assert_array_equal(lagmat, expected)

    def test_lagmat2ds_use_pandas(self):
        data = self.macro_df
        lagmat = stattools.lagmat2ds(data, 2, use_pandas=True)
        expected = self._prepare_expected(data.values, 2)
        cols = []
        for c in data:
            for lags in range(3):
                if lags == 0:
                    cols.append(c)
                else:
                    cols.append(c + '.L.' + str(lags))
        expected = pd.DataFrame(expected, index=data.index, columns=cols)
        assert_frame_equal(lagmat, expected)
        lagmat = stattools.lagmat2ds(data.iloc[:, :2], 3, use_pandas=True, trim='both')
        expected = self._prepare_expected(data.values[:, :2], 3)
        cols = []
        for c in data.iloc[:, :2]:
            for lags in range(4):
                if lags == 0:
                    cols.append(c)
                else:
                    cols.append(c + '.L.' + str(lags))
        expected = pd.DataFrame(expected, index=data.index, columns=cols)
        expected = expected.iloc[3:]
        assert_frame_equal(lagmat, expected)
        data = self.series
        lagmat = stattools.lagmat2ds(data, 5, use_pandas=True)
        expected = self._prepare_expected(data.values[:, None], 5)
        cols = []
        c = data.name
        for lags in range(6):
            if lags == 0:
                cols.append(c)
            else:
                cols.append(c + '.L.' + str(lags))
        expected = pd.DataFrame(expected, index=data.index, columns=cols)
        assert_frame_equal(lagmat, expected)

    def test_3d_error(self):
        data = np.array(2)
        with pytest.raises(ValueError):
            stattools.lagmat2ds(data, 5)
        data = np.zeros((100, 2, 2))
        with pytest.raises(ValueError):
            stattools.lagmat2ds(data, 5)