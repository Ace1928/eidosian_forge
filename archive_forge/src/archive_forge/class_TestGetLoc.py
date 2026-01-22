from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
class TestGetLoc:

    def test_get_loc_key_unit_mismatch(self):
        idx = date_range('2000-01-01', periods=3)
        key = idx[1].as_unit('ms')
        loc = idx.get_loc(key)
        assert loc == 1
        assert key in idx

    def test_get_loc_key_unit_mismatch_not_castable(self):
        dta = date_range('2000-01-01', periods=3)._data.astype('M8[s]')
        dti = DatetimeIndex(dta)
        key = dta[0].as_unit('ns') + pd.Timedelta(1)
        with pytest.raises(KeyError, match="Timestamp\\('2000-01-01 00:00:00.000000001'\\)"):
            dti.get_loc(key)
        assert key not in dti

    def test_get_loc_time_obj(self):
        idx = date_range('2000-01-01', periods=24, freq='h')
        result = idx.get_loc(time(12))
        expected = np.array([12])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)
        result = idx.get_loc(time(12, 30))
        expected = np.array([])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize('offset', [-10, 10])
    def test_get_loc_time_obj2(self, monkeypatch, offset):
        size_cutoff = 50
        n = size_cutoff + offset
        key = time(15, 11, 30)
        start = key.hour * 3600 + key.minute * 60 + key.second
        step = 24 * 3600
        with monkeypatch.context():
            monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
            idx = date_range('2014-11-26', periods=n, freq='s')
            ts = pd.Series(np.random.default_rng(2).standard_normal(n), index=idx)
            locs = np.arange(start, n, step, dtype=np.intp)
            result = ts.index.get_loc(key)
            tm.assert_numpy_array_equal(result, locs)
            tm.assert_series_equal(ts[key], ts.iloc[locs])
            left, right = (ts.copy(), ts.copy())
            left[key] *= -10
            right.iloc[locs] *= -10
            tm.assert_series_equal(left, right)

    def test_get_loc_time_nat(self):
        tic = time(minute=12, second=43, microsecond=145224)
        dti = DatetimeIndex([pd.NaT])
        loc = dti.get_loc(tic)
        expected = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(loc, expected)

    def test_get_loc_nat(self):
        index = DatetimeIndex(['1/3/2000', 'NaT'])
        assert index.get_loc(pd.NaT) == 1
        assert index.get_loc(None) == 1
        assert index.get_loc(np.nan) == 1
        assert index.get_loc(pd.NA) == 1
        assert index.get_loc(np.datetime64('NaT')) == 1
        with pytest.raises(KeyError, match='NaT'):
            index.get_loc(np.timedelta64('NaT'))

    @pytest.mark.parametrize('key', [pd.Timedelta(0), pd.Timedelta(1), timedelta(0)])
    def test_get_loc_timedelta_invalid_key(self, key):
        dti = date_range('1970-01-01', periods=10)
        msg = 'Cannot index DatetimeIndex with [Tt]imedelta'
        with pytest.raises(TypeError, match=msg):
            dti.get_loc(key)

    def test_get_loc_reasonable_key_error(self):
        index = DatetimeIndex(['1/3/2000'])
        with pytest.raises(KeyError, match='2000'):
            index.get_loc('1/1/2000')

    def test_get_loc_year_str(self):
        rng = date_range('1/1/2000', '1/1/2010')
        result = rng.get_loc('2009')
        expected = slice(3288, 3653)
        assert result == expected