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
class TestDatetimeTZDtype(Base):

    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestDatetimeTZDtype
        """
        return DatetimeTZDtype('ns', 'US/Eastern')

    def test_alias_to_unit_raises(self):
        with pytest.raises(ValueError, match='Passing a dtype alias'):
            DatetimeTZDtype('datetime64[ns, US/Central]')

    def test_alias_to_unit_bad_alias_raises(self):
        with pytest.raises(TypeError, match=''):
            DatetimeTZDtype('this is a bad string')
        with pytest.raises(TypeError, match=''):
            DatetimeTZDtype('datetime64[ns, US/NotATZ]')

    def test_hash_vs_equality(self, dtype):
        dtype2 = DatetimeTZDtype('ns', 'US/Eastern')
        dtype3 = DatetimeTZDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)
        dtype4 = DatetimeTZDtype('ns', 'US/Central')
        assert dtype2 != dtype4
        assert hash(dtype2) != hash(dtype4)

    def test_construction_non_nanosecond(self):
        res = DatetimeTZDtype('ms', 'US/Eastern')
        assert res.unit == 'ms'
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
        assert res.str == '|M8[ms]'
        assert str(res) == 'datetime64[ms, US/Eastern]'
        assert res.base == np.dtype('M8[ms]')

    def test_day_not_supported(self):
        msg = 'DatetimeTZDtype only supports s, ms, us, ns units'
        with pytest.raises(ValueError, match=msg):
            DatetimeTZDtype('D', 'US/Eastern')

    def test_subclass(self):
        a = DatetimeTZDtype.construct_from_string('datetime64[ns, US/Eastern]')
        b = DatetimeTZDtype.construct_from_string('datetime64[ns, CET]')
        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_compat(self, dtype):
        msg = 'is_datetime64tz_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
            assert is_datetime64tz_dtype('datetime64[ns, US/Eastern]')
        assert is_datetime64_any_dtype(dtype)
        assert is_datetime64_any_dtype('datetime64[ns, US/Eastern]')
        assert is_datetime64_ns_dtype(dtype)
        assert is_datetime64_ns_dtype('datetime64[ns, US/Eastern]')
        assert not is_datetime64_dtype(dtype)
        assert not is_datetime64_dtype('datetime64[ns, US/Eastern]')

    def test_construction_from_string(self, dtype):
        result = DatetimeTZDtype.construct_from_string('datetime64[ns, US/Eastern]')
        assert is_dtype_equal(dtype, result)

    @pytest.mark.parametrize('string', ['foo', 'datetime64[ns, notatz]', 'datetime64[ps, UTC]', 'datetime64[ns, dateutil/invalid]'])
    def test_construct_from_string_invalid_raises(self, string):
        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        with pytest.raises(TypeError, match=re.escape(msg)):
            DatetimeTZDtype.construct_from_string(string)

    def test_construct_from_string_wrong_type_raises(self):
        msg = "'construct_from_string' expects a string, got <class 'list'>"
        with pytest.raises(TypeError, match=msg):
            DatetimeTZDtype.construct_from_string(['datetime64[ns, notatz]'])

    def test_is_dtype(self, dtype):
        assert not DatetimeTZDtype.is_dtype(None)
        assert DatetimeTZDtype.is_dtype(dtype)
        assert DatetimeTZDtype.is_dtype('datetime64[ns, US/Eastern]')
        assert DatetimeTZDtype.is_dtype('M8[ns, US/Eastern]')
        assert not DatetimeTZDtype.is_dtype('foo')
        assert DatetimeTZDtype.is_dtype(DatetimeTZDtype('ns', 'US/Pacific'))
        assert not DatetimeTZDtype.is_dtype(np.float64)

    def test_equality(self, dtype):
        assert is_dtype_equal(dtype, 'datetime64[ns, US/Eastern]')
        assert is_dtype_equal(dtype, 'M8[ns, US/Eastern]')
        assert is_dtype_equal(dtype, DatetimeTZDtype('ns', 'US/Eastern'))
        assert not is_dtype_equal(dtype, 'foo')
        assert not is_dtype_equal(dtype, DatetimeTZDtype('ns', 'CET'))
        assert not is_dtype_equal(DatetimeTZDtype('ns', 'US/Eastern'), DatetimeTZDtype('ns', 'US/Pacific'))
        assert is_dtype_equal(np.dtype('M8[ns]'), 'datetime64[ns]')
        assert dtype == 'M8[ns, US/Eastern]'

    def test_basic(self, dtype):
        msg = 'is_datetime64tz_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
        dr = date_range('20130101', periods=3, tz='US/Eastern')
        s = Series(dr, name='A')
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(s.dtype)
            assert is_datetime64tz_dtype(s)
            assert not is_datetime64tz_dtype(np.dtype('float64'))
            assert not is_datetime64tz_dtype(1.0)

    def test_dst(self):
        dr1 = date_range('2013-01-01', periods=3, tz='US/Eastern')
        s1 = Series(dr1, name='A')
        assert isinstance(s1.dtype, DatetimeTZDtype)
        dr2 = date_range('2013-08-01', periods=3, tz='US/Eastern')
        s2 = Series(dr2, name='A')
        assert isinstance(s2.dtype, DatetimeTZDtype)
        assert s1.dtype == s2.dtype

    @pytest.mark.parametrize('tz', ['UTC', 'US/Eastern'])
    @pytest.mark.parametrize('constructor', ['M8', 'datetime64'])
    def test_parser(self, tz, constructor):
        dtz_str = f'{constructor}[ns, {tz}]'
        result = DatetimeTZDtype.construct_from_string(dtz_str)
        expected = DatetimeTZDtype('ns', tz)
        assert result == expected

    def test_empty(self):
        with pytest.raises(TypeError, match="A 'tz' is required."):
            DatetimeTZDtype()

    def test_tz_standardize(self):
        tz = pytz.timezone('US/Eastern')
        dr = date_range('2013-01-01', periods=3, tz='US/Eastern')
        dtype = DatetimeTZDtype('ns', dr.tz)
        assert dtype.tz == tz
        dtype = DatetimeTZDtype('ns', dr[0].tz)
        assert dtype.tz == tz