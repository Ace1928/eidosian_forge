import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
class TestIntervalDtype(Base):

    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestIntervalDtype
        """
        return IntervalDtype('int64', 'right')

    def test_hash_vs_equality(self, dtype):
        dtype2 = IntervalDtype('int64', 'right')
        dtype3 = IntervalDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert dtype is not dtype2
        assert dtype2 is not dtype3
        assert dtype3 is not dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)
        dtype1 = IntervalDtype('interval')
        dtype2 = IntervalDtype(dtype1)
        dtype3 = IntervalDtype('interval')
        assert dtype2 == dtype1
        assert dtype2 == dtype2
        assert dtype2 == dtype3
        assert dtype2 is not dtype1
        assert dtype2 is dtype2
        assert dtype2 is not dtype3
        assert hash(dtype2) == hash(dtype1)
        assert hash(dtype2) == hash(dtype2)
        assert hash(dtype2) == hash(dtype3)

    @pytest.mark.parametrize('subtype', ['interval[int64]', 'Interval[int64]', 'int64', np.dtype('int64')])
    def test_construction(self, subtype):
        i = IntervalDtype(subtype, closed='right')
        assert i.subtype == np.dtype('int64')
        msg = 'is_interval_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype(i)

    @pytest.mark.parametrize('subtype', ['interval[int64]', 'Interval[int64]', 'int64', np.dtype('int64')])
    def test_construction_allows_closed_none(self, subtype):
        dtype = IntervalDtype(subtype)
        assert dtype.closed is None

    def test_closed_mismatch(self):
        msg = "'closed' keyword does not match value specified in dtype string"
        with pytest.raises(ValueError, match=msg):
            IntervalDtype('interval[int64, left]', 'right')

    @pytest.mark.parametrize('subtype', [None, 'interval', 'Interval'])
    def test_construction_generic(self, subtype):
        i = IntervalDtype(subtype)
        assert i.subtype is None
        msg = 'is_interval_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype(i)

    @pytest.mark.parametrize('subtype', [CategoricalDtype(list('abc'), False), CategoricalDtype(list('wxyz'), True), object, str, '<U10', 'interval[category]', 'interval[object]'])
    def test_construction_not_supported(self, subtype):
        msg = 'category, object, and string subtypes are not supported for IntervalDtype'
        with pytest.raises(TypeError, match=msg):
            IntervalDtype(subtype)

    @pytest.mark.parametrize('subtype', ['xx', 'IntervalA', 'Interval[foo]'])
    def test_construction_errors(self, subtype):
        msg = 'could not construct IntervalDtype'
        with pytest.raises(TypeError, match=msg):
            IntervalDtype(subtype)

    def test_closed_must_match(self):
        dtype = IntervalDtype(np.float64, 'left')
        msg = "dtype.closed and 'closed' do not match"
        with pytest.raises(ValueError, match=msg):
            IntervalDtype(dtype, closed='both')

    def test_closed_invalid(self):
        with pytest.raises(ValueError, match='closed must be one of'):
            IntervalDtype(np.float64, 'foo')

    def test_construction_from_string(self, dtype):
        result = IntervalDtype('interval[int64, right]')
        assert is_dtype_equal(dtype, result)
        result = IntervalDtype.construct_from_string('interval[int64, right]')
        assert is_dtype_equal(dtype, result)

    @pytest.mark.parametrize('string', [0, 3.14, ('a', 'b'), None])
    def test_construction_from_string_errors(self, string):
        msg = f"'construct_from_string' expects a string, got {type(string)}"
        with pytest.raises(TypeError, match=re.escape(msg)):
            IntervalDtype.construct_from_string(string)

    @pytest.mark.parametrize('string', ['foo', 'foo[int64]', 'IntervalA'])
    def test_construction_from_string_error_subtype(self, string):
        msg = 'Incorrectly formatted string passed to constructor. Valid formats include Interval or Interval\\[dtype\\] where dtype is numeric, datetime, or timedelta'
        with pytest.raises(TypeError, match=msg):
            IntervalDtype.construct_from_string(string)

    def test_subclass(self):
        a = IntervalDtype('interval[int64, right]')
        b = IntervalDtype('interval[int64, right]')
        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_is_dtype(self, dtype):
        assert IntervalDtype.is_dtype(dtype)
        assert IntervalDtype.is_dtype('interval')
        assert IntervalDtype.is_dtype(IntervalDtype('float64'))
        assert IntervalDtype.is_dtype(IntervalDtype('int64'))
        assert IntervalDtype.is_dtype(IntervalDtype(np.int64))
        assert IntervalDtype.is_dtype(IntervalDtype('float64', 'left'))
        assert IntervalDtype.is_dtype(IntervalDtype('int64', 'right'))
        assert IntervalDtype.is_dtype(IntervalDtype(np.int64, 'both'))
        assert not IntervalDtype.is_dtype('D')
        assert not IntervalDtype.is_dtype('3D')
        assert not IntervalDtype.is_dtype('us')
        assert not IntervalDtype.is_dtype('S')
        assert not IntervalDtype.is_dtype('foo')
        assert not IntervalDtype.is_dtype('IntervalA')
        assert not IntervalDtype.is_dtype(np.object_)
        assert not IntervalDtype.is_dtype(np.int64)
        assert not IntervalDtype.is_dtype(np.float64)

    def test_equality(self, dtype):
        assert is_dtype_equal(dtype, 'interval[int64, right]')
        assert is_dtype_equal(dtype, IntervalDtype('int64', 'right'))
        assert is_dtype_equal(IntervalDtype('int64', 'right'), IntervalDtype('int64', 'right'))
        assert not is_dtype_equal(dtype, 'interval[int64]')
        assert not is_dtype_equal(dtype, IntervalDtype('int64'))
        assert not is_dtype_equal(IntervalDtype('int64', 'right'), IntervalDtype('int64'))
        assert not is_dtype_equal(dtype, 'int64')
        assert not is_dtype_equal(IntervalDtype('int64', 'neither'), IntervalDtype('float64', 'right'))
        assert not is_dtype_equal(IntervalDtype('int64', 'both'), IntervalDtype('int64', 'left'))
        dtype1 = IntervalDtype('float64', 'left')
        dtype2 = IntervalDtype('datetime64[ns, US/Eastern]', 'left')
        assert dtype1 != dtype2
        assert dtype2 != dtype1

    @pytest.mark.parametrize('subtype', [None, 'interval', 'Interval', 'int64', 'uint64', 'float64', 'complex128', 'datetime64', 'timedelta64', PeriodDtype('Q')])
    def test_equality_generic(self, subtype):
        closed = 'right' if subtype is not None else None
        dtype = IntervalDtype(subtype, closed=closed)
        assert is_dtype_equal(dtype, 'interval')
        assert is_dtype_equal(dtype, IntervalDtype())

    @pytest.mark.parametrize('subtype', ['int64', 'uint64', 'float64', 'complex128', 'datetime64', 'timedelta64', PeriodDtype('Q')])
    def test_name_repr(self, subtype):
        closed = 'right' if subtype is not None else None
        dtype = IntervalDtype(subtype, closed=closed)
        expected = f'interval[{subtype}, {closed}]'
        assert str(dtype) == expected
        assert dtype.name == 'interval'

    @pytest.mark.parametrize('subtype', [None, 'interval', 'Interval'])
    def test_name_repr_generic(self, subtype):
        dtype = IntervalDtype(subtype)
        assert str(dtype) == 'interval'
        assert dtype.name == 'interval'

    def test_basic(self, dtype):
        msg = 'is_interval_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype(dtype)
            ii = IntervalIndex.from_breaks(range(3))
            assert is_interval_dtype(ii.dtype)
            assert is_interval_dtype(ii)
            s = Series(ii, name='A')
            assert is_interval_dtype(s.dtype)
            assert is_interval_dtype(s)

    def test_basic_dtype(self):
        msg = 'is_interval_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype('interval[int64, both]')
            assert is_interval_dtype(IntervalIndex.from_tuples([(0, 1)]))
            assert is_interval_dtype(IntervalIndex.from_breaks(np.arange(4)))
            assert is_interval_dtype(IntervalIndex.from_breaks(date_range('20130101', periods=3)))
            assert not is_interval_dtype('U')
            assert not is_interval_dtype('S')
            assert not is_interval_dtype('foo')
            assert not is_interval_dtype(np.object_)
            assert not is_interval_dtype(np.int64)
            assert not is_interval_dtype(np.float64)

    def test_caching(self):
        IntervalDtype.reset_cache()
        dtype = IntervalDtype('int64', 'right')
        assert len(IntervalDtype._cache_dtypes) == 0
        IntervalDtype('interval')
        assert len(IntervalDtype._cache_dtypes) == 0
        IntervalDtype.reset_cache()
        tm.round_trip_pickle(dtype)
        assert len(IntervalDtype._cache_dtypes) == 0

    def test_not_string(self):
        assert not is_string_dtype(IntervalDtype())

    def test_unpickling_without_closed(self):
        dtype = IntervalDtype('interval')
        assert dtype._closed is None
        tm.round_trip_pickle(dtype)

    def test_dont_keep_ref_after_del(self):
        dtype = IntervalDtype('int64', 'right')
        ref = weakref.ref(dtype)
        del dtype
        assert ref() is None