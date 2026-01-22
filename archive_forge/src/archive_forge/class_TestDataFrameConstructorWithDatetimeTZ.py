import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
class TestDataFrameConstructorWithDatetimeTZ:

    @pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_construction_preserves_tzaware_dtypes(self, tz):
        dr = date_range('2011/1/1', '2012/1/1', freq='W-FRI')
        dr_tz = dr.tz_localize(tz)
        df = DataFrame({'A': 'foo', 'B': dr_tz}, index=dr)
        tz_expected = DatetimeTZDtype('ns', dr_tz.tzinfo)
        assert df['B'].dtype == tz_expected
        datetimes_naive = [ts.to_pydatetime() for ts in dr]
        datetimes_with_tz = [ts.to_pydatetime() for ts in dr_tz]
        df = DataFrame({'dr': dr})
        df['dr_tz'] = dr_tz
        df['datetimes_naive'] = datetimes_naive
        df['datetimes_with_tz'] = datetimes_with_tz
        result = df.dtypes
        expected = Series([np.dtype('datetime64[ns]'), DatetimeTZDtype(tz=tz), np.dtype('datetime64[ns]'), DatetimeTZDtype(tz=tz)], index=['dr', 'dr_tz', 'datetimes_naive', 'datetimes_with_tz'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('pydt', [True, False])
    def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture, pydt):
        tz = tz_aware_fixture
        ts = Timestamp('2019', tz=tz)
        if pydt:
            ts = ts.to_pydatetime()
        msg = 'Cannot convert timezone-aware data to timezone-naive dtype. Use pd.Series\\(values\\).dt.tz_localize\\(None\\) instead.'
        with pytest.raises(ValueError, match=msg):
            DataFrame({0: [ts]}, dtype='datetime64[ns]')
        msg2 = 'Cannot unbox tzaware Timestamp to tznaive dtype'
        with pytest.raises(TypeError, match=msg2):
            DataFrame({0: ts}, index=[0], dtype='datetime64[ns]')
        with pytest.raises(ValueError, match=msg):
            DataFrame([ts], dtype='datetime64[ns]')
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.array([ts], dtype=object), dtype='datetime64[ns]')
        with pytest.raises(TypeError, match=msg2):
            DataFrame(ts, index=[0], columns=[0], dtype='datetime64[ns]')
        with pytest.raises(ValueError, match=msg):
            DataFrame([Series([ts])], dtype='datetime64[ns]')
        with pytest.raises(ValueError, match=msg):
            DataFrame([[ts]], columns=[0], dtype='datetime64[ns]')

    def test_from_dict(self):
        idx = Index(date_range('20130101', periods=3, tz='US/Eastern'), name='foo')
        dr = date_range('20130110', periods=3)
        df = DataFrame({'A': idx, 'B': dr})
        assert df['A'].dtype, 'M8[ns, US/Eastern'
        assert df['A'].name == 'A'
        tm.assert_series_equal(df['A'], Series(idx, name='A'))
        tm.assert_series_equal(df['B'], Series(dr, name='B'))

    def test_from_index(self):
        idx2 = date_range('20130101', periods=3, tz='US/Eastern', name='foo')
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2['foo'], Series(idx2, name='foo'))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2['foo'], Series(idx2, name='foo'))
        idx2 = date_range('20130101', periods=3, tz='US/Eastern')
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2[0], Series(idx2, name=0))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2[0], Series(idx2, name=0))

    def test_frame_dict_constructor_datetime64_1680(self):
        dr = date_range('1/1/2012', periods=10)
        s = Series(dr, index=dr)
        DataFrame({'a': 'foo', 'b': s}, index=dr)
        DataFrame({'a': 'foo', 'b': s.values}, index=dr)

    def test_frame_datetime64_mixed_index_ctor_1681(self):
        dr = date_range('2011/1/1', '2012/1/1', freq='W-FRI')
        ts = Series(dr)
        d = DataFrame({'A': 'foo', 'B': ts}, index=dr)
        assert d['B'].isna().all()

    def test_frame_timeseries_column(self):
        dr = date_range(start='20130101T10:00:00', periods=3, freq='min', tz='US/Eastern')
        result = DataFrame(dr, columns=['timestamps'])
        expected = DataFrame({'timestamps': [Timestamp('20130101T10:00:00', tz='US/Eastern'), Timestamp('20130101T10:01:00', tz='US/Eastern'), Timestamp('20130101T10:02:00', tz='US/Eastern')]})
        tm.assert_frame_equal(result, expected)

    def test_nested_dict_construction(self):
        columns = ['Nevada', 'Ohio']
        pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
        result = DataFrame(pop, index=[2001, 2002, 2003], columns=columns)
        expected = DataFrame([(2.4, 1.7), (2.9, 3.6), (np.nan, np.nan)], columns=columns, index=Index([2001, 2002, 2003]))
        tm.assert_frame_equal(result, expected)

    def test_from_tzaware_object_array(self):
        dti = date_range('2016-04-05 04:30', periods=3, tz='UTC')
        data = dti._data.astype(object).reshape(1, -1)
        df = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == dti.dtype).all()
        assert (df == dti).all().all()

    def test_from_tzaware_mixed_object_array(self):
        arr = np.array([[Timestamp('2013-01-01 00:00:00'), Timestamp('2013-01-02 00:00:00'), Timestamp('2013-01-03 00:00:00')], [Timestamp('2013-01-01 00:00:00-0500', tz='US/Eastern'), pd.NaT, Timestamp('2013-01-03 00:00:00-0500', tz='US/Eastern')], [Timestamp('2013-01-01 00:00:00+0100', tz='CET'), pd.NaT, Timestamp('2013-01-03 00:00:00+0100', tz='CET')]], dtype=object).T
        res = DataFrame(arr, columns=['A', 'B', 'C'])
        expected_dtypes = ['datetime64[ns]', 'datetime64[ns, US/Eastern]', 'datetime64[ns, CET]']
        assert (res.dtypes == expected_dtypes).all()

    def test_from_2d_ndarray_with_dtype(self):
        array_dim2 = np.arange(10).reshape((5, 2))
        df = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('typ', [set, frozenset])
    def test_construction_from_set_raises(self, typ):
        values = typ({1, 2, 3})
        msg = f"'{typ.__name__}' type is unordered"
        with pytest.raises(TypeError, match=msg):
            DataFrame({'a': values})
        with pytest.raises(TypeError, match=msg):
            Series(values)

    def test_construction_from_ndarray_datetimelike(self):
        arr = np.arange(0, 12, dtype='datetime64[ns]').reshape(4, 3)
        df = DataFrame(arr)
        assert all((isinstance(arr, DatetimeArray) for arr in df._mgr.arrays))

    def test_construction_from_ndarray_with_eadtype_mismatched_columns(self):
        arr = np.random.default_rng(2).standard_normal((10, 2))
        dtype = pd.array([2.0]).dtype
        msg = 'len\\(arrays\\) must match len\\(columns\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, columns=['foo'], dtype=dtype)
        arr2 = pd.array([2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr2, columns=['foo', 'bar'])

    def test_columns_indexes_raise_on_sets(self):
        data = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match='index cannot be a set'):
            DataFrame(data, index={'a', 'b'})
        with pytest.raises(ValueError, match='columns cannot be a set'):
            DataFrame(data, columns={'a', 'b', 'c'})