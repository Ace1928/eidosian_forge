from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
class TestIndexConstructorInference:

    def test_object_all_bools(self):
        arr = np.array([True, False], dtype=object)
        res = Index(arr)
        assert res.dtype == object
        assert Series(arr).dtype == object

    def test_object_all_complex(self):
        arr = np.array([complex(1), complex(2)], dtype=object)
        res = Index(arr)
        assert res.dtype == object
        assert Series(arr).dtype == object

    @pytest.mark.parametrize('val', [NaT, None, np.nan, float('nan')])
    def test_infer_nat(self, val):
        values = [NaT, val]
        idx = Index(values)
        assert idx.dtype == 'datetime64[ns]' and idx.isna().all()
        idx = Index(values[::-1])
        assert idx.dtype == 'datetime64[ns]' and idx.isna().all()
        idx = Index(np.array(values, dtype=object))
        assert idx.dtype == 'datetime64[ns]' and idx.isna().all()
        idx = Index(np.array(values, dtype=object)[::-1])
        assert idx.dtype == 'datetime64[ns]' and idx.isna().all()

    @pytest.mark.parametrize('na_value', [None, np.nan])
    @pytest.mark.parametrize('vtype', [list, tuple, iter])
    def test_construction_list_tuples_nan(self, na_value, vtype):
        values = [(1, 'two'), (3.0, na_value)]
        result = Index(vtype(values))
        expected = MultiIndex.from_tuples(values)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dtype', [int, 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8'])
    def test_constructor_int_dtype_float(self, dtype):
        expected = Index([0, 1, 2, 3], dtype=dtype)
        result = Index([0.0, 1.0, 2.0, 3.0], dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('cast_index', [True, False])
    @pytest.mark.parametrize('vals', [[True, False, True], np.array([True, False, True], dtype=bool)])
    def test_constructor_dtypes_to_object(self, cast_index, vals):
        if cast_index:
            index = Index(vals, dtype=bool)
        else:
            index = Index(vals)
        assert type(index) is Index
        assert index.dtype == bool

    def test_constructor_categorical_to_object(self):
        ci = CategoricalIndex(range(5))
        result = Index(ci, dtype=object)
        assert not isinstance(result, CategoricalIndex)

    def test_constructor_infer_periodindex(self):
        xp = period_range('2012-1-1', freq='M', periods=3)
        rs = Index(xp)
        tm.assert_index_equal(rs, xp)
        assert isinstance(rs, PeriodIndex)

    def test_from_list_of_periods(self):
        rng = period_range('1/1/2000', periods=20, freq='D')
        periods = list(rng)
        result = Index(periods)
        assert isinstance(result, PeriodIndex)

    @pytest.mark.parametrize('pos', [0, 1])
    @pytest.mark.parametrize('klass,dtype,ctor', [(DatetimeIndex, 'datetime64[ns]', np.datetime64('nat')), (TimedeltaIndex, 'timedelta64[ns]', np.timedelta64('nat'))])
    def test_constructor_infer_nat_dt_like(self, pos, klass, dtype, ctor, nulls_fixture, request):
        if isinstance(nulls_fixture, Decimal):
            pytest.skip(f"We don't cast {type(nulls_fixture).__name__} to datetime64/timedelta64")
        expected = klass([NaT, NaT])
        assert expected.dtype == dtype
        data = [ctor]
        data.insert(pos, nulls_fixture)
        warn = None
        if nulls_fixture is NA:
            expected = Index([NA, NaT])
            mark = pytest.mark.xfail(reason='Broken with np.NaT ctor; see GH 31884')
            request.applymarker(mark)
            warn = DeprecationWarning
        result = Index(data)
        with tm.assert_produces_warning(warn):
            tm.assert_index_equal(result, expected)
        result = Index(np.array(data, dtype=object))
        with tm.assert_produces_warning(warn):
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('swap_objs', [True, False])
    def test_constructor_mixed_nat_objs_infers_object(self, swap_objs):
        data = [np.datetime64('nat'), np.timedelta64('nat')]
        if swap_objs:
            data = data[::-1]
        expected = Index(data, dtype=object)
        tm.assert_index_equal(Index(data), expected)
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)

    @pytest.mark.parametrize('swap_objs', [True, False])
    def test_constructor_datetime_and_datetime64(self, swap_objs):
        data = [Timestamp(2021, 6, 8, 9, 42), np.datetime64('now')]
        if swap_objs:
            data = data[::-1]
        expected = DatetimeIndex(data)
        tm.assert_index_equal(Index(data), expected)
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)

    def test_constructor_datetimes_mixed_tzs(self):
        tz = maybe_get_tz('US/Central')
        dt1 = datetime(2020, 1, 1, tzinfo=tz)
        dt2 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result = Index([dt1, dt2])
        expected = Index([dt1, dt2], dtype=object)
        tm.assert_index_equal(result, expected)