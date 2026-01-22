import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestSeriesCov:

    def test_cov(self, datetime_series):
        tm.assert_almost_equal(datetime_series.cov(datetime_series), datetime_series.std() ** 2)
        tm.assert_almost_equal(datetime_series[:15].cov(datetime_series[5:]), datetime_series[5:15].std() ** 2)
        assert np.isnan(datetime_series[::2].cov(datetime_series[1::2]))
        cp = datetime_series[:10].copy()
        cp[:] = np.nan
        assert isna(cp.cov(cp))
        assert isna(datetime_series[:15].cov(datetime_series[5:], min_periods=12))
        ts1 = datetime_series[:15].reindex(datetime_series.index)
        ts2 = datetime_series[5:].reindex(datetime_series.index)
        assert isna(ts1.cov(ts2, min_periods=12))

    @pytest.mark.parametrize('test_ddof', [None, 0, 1, 2, 3])
    @pytest.mark.parametrize('dtype', ['float64', 'Float64'])
    def test_cov_ddof(self, test_ddof, dtype):
        np_array1 = np.random.default_rng(2).random(10)
        np_array2 = np.random.default_rng(2).random(10)
        s1 = Series(np_array1, dtype=dtype)
        s2 = Series(np_array2, dtype=dtype)
        result = s1.cov(s2, ddof=test_ddof)
        expected = np.cov(np_array1, np_array2, ddof=test_ddof)[0][1]
        assert math.isclose(expected, result)