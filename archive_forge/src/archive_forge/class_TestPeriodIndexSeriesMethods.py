import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
class TestPeriodIndexSeriesMethods:
    """Test PeriodIndex and Period Series Ops consistency"""

    def _check(self, values, func, expected):
        idx = PeriodIndex(values)
        result = func(idx)
        tm.assert_equal(result, expected)
        ser = Series(values)
        result = func(ser)
        exp = Series(expected, name=values.name)
        tm.assert_series_equal(result, exp)

    def test_pi_ops(self):
        idx = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq='M', name='idx')
        expected = PeriodIndex(['2011-03', '2011-04', '2011-05', '2011-06'], freq='M', name='idx')
        self._check(idx, lambda x: x + 2, expected)
        self._check(idx, lambda x: 2 + x, expected)
        self._check(idx + 2, lambda x: x - 2, idx)
        result = idx - Period('2011-01', freq='M')
        off = idx.freq
        exp = pd.Index([0 * off, 1 * off, 2 * off, 3 * off], name='idx')
        tm.assert_index_equal(result, exp)
        result = Period('2011-01', freq='M') - idx
        exp = pd.Index([0 * off, -1 * off, -2 * off, -3 * off], name='idx')
        tm.assert_index_equal(result, exp)

    @pytest.mark.parametrize('ng', ['str', 1.5])
    @pytest.mark.parametrize('func', [lambda obj, ng: obj + ng, lambda obj, ng: ng + obj, lambda obj, ng: obj - ng, lambda obj, ng: ng - obj, lambda obj, ng: np.add(obj, ng), lambda obj, ng: np.add(ng, obj), lambda obj, ng: np.subtract(obj, ng), lambda obj, ng: np.subtract(ng, obj)])
    def test_parr_ops_errors(self, ng, func, box_with_array):
        idx = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq='M', name='idx')
        obj = tm.box_expected(idx, box_with_array)
        msg = '|'.join(['unsupported operand type\\(s\\)', 'can only concatenate', 'must be str', 'object to str implicitly'])
        with pytest.raises(TypeError, match=msg):
            func(obj, ng)

    def test_pi_ops_nat(self):
        idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-04'], freq='M', name='idx')
        expected = PeriodIndex(['2011-03', '2011-04', 'NaT', '2011-06'], freq='M', name='idx')
        self._check(idx, lambda x: x + 2, expected)
        self._check(idx, lambda x: 2 + x, expected)
        self._check(idx, lambda x: np.add(x, 2), expected)
        self._check(idx + 2, lambda x: x - 2, idx)
        self._check(idx + 2, lambda x: np.subtract(x, 2), idx)
        idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-04'], freq='2M', name='idx')
        expected = PeriodIndex(['2011-07', '2011-08', 'NaT', '2011-10'], freq='2M', name='idx')
        self._check(idx, lambda x: x + 3, expected)
        self._check(idx, lambda x: 3 + x, expected)
        self._check(idx, lambda x: np.add(x, 3), expected)
        self._check(idx + 3, lambda x: x - 3, idx)
        self._check(idx + 3, lambda x: np.subtract(x, 3), idx)

    def test_pi_ops_array_int(self):
        idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-04'], freq='M', name='idx')
        f = lambda x: x + np.array([1, 2, 3, 4])
        exp = PeriodIndex(['2011-02', '2011-04', 'NaT', '2011-08'], freq='M', name='idx')
        self._check(idx, f, exp)
        f = lambda x: np.add(x, np.array([4, -1, 1, 2]))
        exp = PeriodIndex(['2011-05', '2011-01', 'NaT', '2011-06'], freq='M', name='idx')
        self._check(idx, f, exp)
        f = lambda x: x - np.array([1, 2, 3, 4])
        exp = PeriodIndex(['2010-12', '2010-12', 'NaT', '2010-12'], freq='M', name='idx')
        self._check(idx, f, exp)
        f = lambda x: np.subtract(x, np.array([3, 2, 3, -2]))
        exp = PeriodIndex(['2010-10', '2010-12', 'NaT', '2011-06'], freq='M', name='idx')
        self._check(idx, f, exp)

    def test_pi_ops_offset(self):
        idx = PeriodIndex(['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01'], freq='D', name='idx')
        f = lambda x: x + pd.offsets.Day()
        exp = PeriodIndex(['2011-01-02', '2011-02-02', '2011-03-02', '2011-04-02'], freq='D', name='idx')
        self._check(idx, f, exp)
        f = lambda x: x + pd.offsets.Day(2)
        exp = PeriodIndex(['2011-01-03', '2011-02-03', '2011-03-03', '2011-04-03'], freq='D', name='idx')
        self._check(idx, f, exp)
        f = lambda x: x - pd.offsets.Day(2)
        exp = PeriodIndex(['2010-12-30', '2011-01-30', '2011-02-27', '2011-03-30'], freq='D', name='idx')
        self._check(idx, f, exp)

    def test_pi_offset_errors(self):
        idx = PeriodIndex(['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01'], freq='D', name='idx')
        ser = Series(idx)
        msg = "Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq"
        for obj in [idx, ser]:
            with pytest.raises(IncompatibleFrequency, match=msg):
                obj + pd.offsets.Hour(2)
            with pytest.raises(IncompatibleFrequency, match=msg):
                pd.offsets.Hour(2) + obj
            with pytest.raises(IncompatibleFrequency, match=msg):
                obj - pd.offsets.Hour(2)

    def test_pi_sub_period(self):
        idx = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq='M', name='idx')
        result = idx - Period('2012-01', freq='M')
        off = idx.freq
        exp = pd.Index([-12 * off, -11 * off, -10 * off, -9 * off], name='idx')
        tm.assert_index_equal(result, exp)
        result = np.subtract(idx, Period('2012-01', freq='M'))
        tm.assert_index_equal(result, exp)
        result = Period('2012-01', freq='M') - idx
        exp = pd.Index([12 * off, 11 * off, 10 * off, 9 * off], name='idx')
        tm.assert_index_equal(result, exp)
        result = np.subtract(Period('2012-01', freq='M'), idx)
        tm.assert_index_equal(result, exp)
        exp = TimedeltaIndex([np.nan, np.nan, np.nan, np.nan], name='idx')
        result = idx - Period('NaT', freq='M')
        tm.assert_index_equal(result, exp)
        assert result.freq == exp.freq
        result = Period('NaT', freq='M') - idx
        tm.assert_index_equal(result, exp)
        assert result.freq == exp.freq

    def test_pi_sub_pdnat(self):
        idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-04'], freq='M', name='idx')
        exp = TimedeltaIndex([pd.NaT] * 4, name='idx')
        tm.assert_index_equal(pd.NaT - idx, exp)
        tm.assert_index_equal(idx - pd.NaT, exp)

    def test_pi_sub_period_nat(self):
        idx = PeriodIndex(['2011-01', 'NaT', '2011-03', '2011-04'], freq='M', name='idx')
        result = idx - Period('2012-01', freq='M')
        off = idx.freq
        exp = pd.Index([-12 * off, pd.NaT, -10 * off, -9 * off], name='idx')
        tm.assert_index_equal(result, exp)
        result = Period('2012-01', freq='M') - idx
        exp = pd.Index([12 * off, pd.NaT, 10 * off, 9 * off], name='idx')
        tm.assert_index_equal(result, exp)
        exp = TimedeltaIndex([np.nan, np.nan, np.nan, np.nan], name='idx')
        tm.assert_index_equal(idx - Period('NaT', freq='M'), exp)
        tm.assert_index_equal(Period('NaT', freq='M') - idx, exp)