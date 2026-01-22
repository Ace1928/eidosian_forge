from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
class TestDtypeEnforced:

    def test_constructor_object_dtype_with_ea_data(self, any_numeric_ea_dtype):
        arr = array([0], dtype=any_numeric_ea_dtype)
        idx = Index(arr, dtype=object)
        assert idx.dtype == object

    @pytest.mark.parametrize('dtype', [object, 'float64', 'uint64', 'category'])
    def test_constructor_range_values_mismatched_dtype(self, dtype):
        rng = Index(range(5))
        result = Index(rng, dtype=dtype)
        assert result.dtype == dtype
        result = Index(range(5), dtype=dtype)
        assert result.dtype == dtype

    @pytest.mark.parametrize('dtype', [object, 'float64', 'uint64', 'category'])
    def test_constructor_categorical_values_mismatched_non_ea_dtype(self, dtype):
        cat = Categorical([1, 2, 3])
        result = Index(cat, dtype=dtype)
        assert result.dtype == dtype

    def test_constructor_categorical_values_mismatched_dtype(self):
        dti = date_range('2016-01-01', periods=3)
        cat = Categorical(dti)
        result = Index(cat, dti.dtype)
        tm.assert_index_equal(result, dti)
        dti2 = dti.tz_localize('Asia/Tokyo')
        cat2 = Categorical(dti2)
        result = Index(cat2, dti2.dtype)
        tm.assert_index_equal(result, dti2)
        ii = IntervalIndex.from_breaks(range(5))
        cat3 = Categorical(ii)
        result = Index(cat3, dtype=ii.dtype)
        tm.assert_index_equal(result, ii)

    def test_constructor_ea_values_mismatched_categorical_dtype(self):
        dti = date_range('2016-01-01', periods=3)
        result = Index(dti, dtype='category')
        expected = CategoricalIndex(dti)
        tm.assert_index_equal(result, expected)
        dti2 = date_range('2016-01-01', periods=3, tz='US/Pacific')
        result = Index(dti2, dtype='category')
        expected = CategoricalIndex(dti2)
        tm.assert_index_equal(result, expected)

    def test_constructor_period_values_mismatched_dtype(self):
        pi = period_range('2016-01-01', periods=3, freq='D')
        result = Index(pi, dtype='category')
        expected = CategoricalIndex(pi)
        tm.assert_index_equal(result, expected)

    def test_constructor_timedelta64_values_mismatched_dtype(self):
        tdi = timedelta_range('4 Days', periods=5)
        result = Index(tdi, dtype='category')
        expected = CategoricalIndex(tdi)
        tm.assert_index_equal(result, expected)

    def test_constructor_interval_values_mismatched_dtype(self):
        dti = date_range('2016-01-01', periods=3)
        ii = IntervalIndex.from_breaks(dti)
        result = Index(ii, dtype='category')
        expected = CategoricalIndex(ii)
        tm.assert_index_equal(result, expected)

    def test_constructor_datetime64_values_mismatched_period_dtype(self):
        dti = date_range('2016-01-01', periods=3)
        result = Index(dti, dtype='Period[D]')
        expected = dti.to_period('D')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['int64', 'uint64'])
    def test_constructor_int_dtype_nan_raises(self, dtype):
        data = [np.nan]
        msg = 'cannot convert'
        with pytest.raises(ValueError, match=msg):
            Index(data, dtype=dtype)

    @pytest.mark.parametrize('vals', [[1, 2, 3], np.array([1, 2, 3]), np.array([1, 2, 3], dtype=int), [1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0], dtype=float)])
    def test_constructor_dtypes_to_int(self, vals, any_int_numpy_dtype):
        dtype = any_int_numpy_dtype
        index = Index(vals, dtype=dtype)
        assert index.dtype == dtype

    @pytest.mark.parametrize('vals', [[1, 2, 3], [1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3], dtype=int), np.array([1.0, 2.0, 3.0], dtype=float)])
    def test_constructor_dtypes_to_float(self, vals, float_numpy_dtype):
        dtype = float_numpy_dtype
        index = Index(vals, dtype=dtype)
        assert index.dtype == dtype

    @pytest.mark.parametrize('vals', [[1, 2, 3], np.array([1, 2, 3], dtype=int), np.array(['2011-01-01', '2011-01-02'], dtype='datetime64[ns]'), [datetime(2011, 1, 1), datetime(2011, 1, 2)]])
    def test_constructor_dtypes_to_categorical(self, vals):
        index = Index(vals, dtype='category')
        assert isinstance(index, CategoricalIndex)

    @pytest.mark.parametrize('cast_index', [True, False])
    @pytest.mark.parametrize('vals', [Index(np.array([np.datetime64('2011-01-01'), np.datetime64('2011-01-02')])), Index([datetime(2011, 1, 1), datetime(2011, 1, 2)])])
    def test_constructor_dtypes_to_datetime(self, cast_index, vals):
        if cast_index:
            index = Index(vals, dtype=object)
            assert isinstance(index, Index)
            assert index.dtype == object
        else:
            index = Index(vals)
            assert isinstance(index, DatetimeIndex)

    @pytest.mark.parametrize('cast_index', [True, False])
    @pytest.mark.parametrize('vals', [np.array([np.timedelta64(1, 'D'), np.timedelta64(1, 'D')]), [timedelta(1), timedelta(1)]])
    def test_constructor_dtypes_to_timedelta(self, cast_index, vals):
        if cast_index:
            index = Index(vals, dtype=object)
            assert isinstance(index, Index)
            assert index.dtype == object
        else:
            index = Index(vals)
            assert isinstance(index, TimedeltaIndex)

    def test_pass_timedeltaindex_to_index(self):
        rng = timedelta_range('1 days', '10 days')
        idx = Index(rng, dtype=object)
        expected = Index(rng.to_pytimedelta(), dtype=object)
        tm.assert_numpy_array_equal(idx.values, expected.values)

    def test_pass_datetimeindex_to_index(self):
        rng = date_range('1/1/2000', '3/1/2000')
        idx = Index(rng, dtype=object)
        expected = Index(rng.to_pydatetime(), dtype=object)
        tm.assert_numpy_array_equal(idx.values, expected.values)