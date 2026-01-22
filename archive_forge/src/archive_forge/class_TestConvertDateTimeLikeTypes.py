import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
class TestConvertDateTimeLikeTypes:
    """
    Conversion tests for datetime- and timestamp-like types (date64, etc.).
    """

    def test_timestamps_notimezone_no_nulls(self):
        df = pd.DataFrame({'datetime64': np.array(['2007-07-13T01:23:34.123456789', '2006-01-13T12:34:56.432539784', '2010-08-13T05:46:57.437699912'], dtype='datetime64[ns]')})
        field = pa.field('datetime64', pa.timestamp('ns'))
        schema = pa.schema([field])
        _check_pandas_roundtrip(df, expected_schema=schema)

    def test_timestamps_notimezone_nulls(self):
        df = pd.DataFrame({'datetime64': np.array(['2007-07-13T01:23:34.123456789', None, '2010-08-13T05:46:57.437699912'], dtype='datetime64[ns]')})
        field = pa.field('datetime64', pa.timestamp('ns'))
        schema = pa.schema([field])
        _check_pandas_roundtrip(df, expected_schema=schema)

    @pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
    def test_timestamps_with_timezone(self, unit):
        if Version(pd.__version__) < Version('2.0.0') and unit != 'ns':
            pytest.skip('pandas < 2.0 only supports nanosecond datetime64')
        df = pd.DataFrame({'datetime64': np.array(['2007-07-13T01:23:34.123', '2006-01-13T12:34:56.432', '2010-08-13T05:46:57.437'], dtype=f'datetime64[{unit}]')})
        df['datetime64'] = df['datetime64'].dt.tz_localize('US/Eastern')
        _check_pandas_roundtrip(df)
        _check_series_roundtrip(df['datetime64'])
        df = pd.DataFrame({'datetime64': np.array(['2007-07-13T01:23:34.123456789', None, '2006-01-13T12:34:56.432539784', '2010-08-13T05:46:57.437699912'], dtype=f'datetime64[{unit}]')})
        df['datetime64'] = df['datetime64'].dt.tz_localize('US/Eastern')
        _check_pandas_roundtrip(df)

    def test_python_datetime(self):
        date_array = [datetime.today() + timedelta(days=x) for x in range(10)]
        df = pd.DataFrame({'datetime': pd.Series(date_array, dtype=object)})
        table = pa.Table.from_pandas(df)
        assert isinstance(table[0].chunk(0), pa.TimestampArray)
        result = table.to_pandas()
        expected_df = pd.DataFrame({'datetime': pd.Series(date_array, dtype='datetime64[us]')})
        tm.assert_frame_equal(expected_df, result)

    def test_python_datetime_with_pytz_tzinfo(self):
        pytz = pytest.importorskip('pytz')
        for tz in [pytz.utc, pytz.timezone('US/Eastern'), pytz.FixedOffset(1)]:
            values = [datetime(2018, 1, 1, 12, 23, 45, tzinfo=tz)]
            df = pd.DataFrame({'datetime': values})
            _check_pandas_roundtrip(df)

    @h.given(st.none() | past.timezones)
    @h.settings(deadline=None)
    def test_python_datetime_with_pytz_timezone(self, tz):
        if str(tz) in ['build/etc/localtime', 'Factory']:
            pytest.skip('Localtime timezone not supported')
        values = [datetime(2018, 1, 1, 12, 23, 45, tzinfo=tz)]
        df = pd.DataFrame({'datetime': values})
        _check_pandas_roundtrip(df, check_dtype=False)

    def test_python_datetime_with_timezone_tzinfo(self):
        pytz = pytest.importorskip('pytz')
        from datetime import timezone
        values = [datetime(2018, 1, 1, 12, 23, 45, tzinfo=timezone.utc)]
        df = pd.DataFrame({'datetime': values}, index=values)
        _check_pandas_roundtrip(df, preserve_index=True)
        hours = 1
        tz_timezone = timezone(timedelta(hours=hours))
        tz_pytz = pytz.FixedOffset(hours * 60)
        values = [datetime(2018, 1, 1, 12, 23, 45, tzinfo=tz_timezone)]
        values_exp = [datetime(2018, 1, 1, 12, 23, 45, tzinfo=tz_pytz)]
        df = pd.DataFrame({'datetime': values}, index=values)
        df_exp = pd.DataFrame({'datetime': values_exp}, index=values_exp)
        _check_pandas_roundtrip(df, expected=df_exp, preserve_index=True)

    def test_python_datetime_subclass(self):

        class MyDatetime(datetime):
            nanosecond = 0.0
        date_array = [MyDatetime(2000, 1, 1, 1, 1, 1)]
        df = pd.DataFrame({'datetime': pd.Series(date_array, dtype=object)})
        table = pa.Table.from_pandas(df)
        assert isinstance(table[0].chunk(0), pa.TimestampArray)
        result = table.to_pandas()
        expected_df = pd.DataFrame({'datetime': pd.Series(date_array, dtype='datetime64[us]')})
        expected_df['datetime'] = pd.to_datetime(expected_df['datetime'])
        tm.assert_frame_equal(expected_df, result)

    def test_python_date_subclass(self):

        class MyDate(date):
            pass
        date_array = [MyDate(2000, 1, 1)]
        df = pd.DataFrame({'date': pd.Series(date_array, dtype=object)})
        table = pa.Table.from_pandas(df)
        assert isinstance(table[0].chunk(0), pa.Date32Array)
        result = table.to_pandas()
        expected_df = pd.DataFrame({'date': np.array([date(2000, 1, 1)], dtype=object)})
        tm.assert_frame_equal(expected_df, result)

    def test_datetime64_to_date32(self):
        arr = pa.array([date(2017, 10, 23), None])
        c = pa.chunked_array([arr])
        s = c.to_pandas()
        arr2 = pa.Array.from_pandas(s, type=pa.date32())
        assert arr2.equals(arr.cast('date32'))

    @pytest.mark.parametrize('mask', [None, np.array([True, False, False, True, False, False])])
    def test_pandas_datetime_to_date64(self, mask):
        s = pd.to_datetime(['2018-05-10T00:00:00', '2018-05-11T00:00:00', '2018-05-12T00:00:00', '2018-05-10T10:24:01', '2018-05-11T10:24:01', '2018-05-12T10:24:01'])
        arr = pa.Array.from_pandas(s, type=pa.date64(), mask=mask)
        data = np.array([date(2018, 5, 10), date(2018, 5, 11), date(2018, 5, 12), date(2018, 5, 10), date(2018, 5, 11), date(2018, 5, 12)])
        expected = pa.array(data, mask=mask, type=pa.date64())
        assert arr.equals(expected)

    @pytest.mark.parametrize('coerce_to_ns,expected_dtype', [(False, 'datetime64[ms]'), (True, 'datetime64[ns]')])
    def test_array_types_date_as_object(self, coerce_to_ns, expected_dtype):
        data = [date(2000, 1, 1), None, date(1970, 1, 1), date(2040, 2, 26)]
        expected_days = np.array(['2000-01-01', None, '1970-01-01', '2040-02-26'], dtype='datetime64[D]')
        if Version(pd.__version__) < Version('2.0.0'):
            expected_dtype = 'datetime64[ns]'
        expected = np.array(['2000-01-01', None, '1970-01-01', '2040-02-26'], dtype=expected_dtype)
        objects = [pa.array(data), pa.chunked_array([data])]
        for obj in objects:
            result = obj.to_pandas(coerce_temporal_nanoseconds=coerce_to_ns)
            expected_obj = expected_days.astype(object)
            assert result.dtype == expected_obj.dtype
            npt.assert_array_equal(result, expected_obj)
            result = obj.to_pandas(date_as_object=False, coerce_temporal_nanoseconds=coerce_to_ns)
            assert result.dtype == expected.dtype
            npt.assert_array_equal(result, expected)

    @pytest.mark.parametrize('coerce_to_ns,expected_type', [(False, 'datetime64[ms]'), (True, 'datetime64[ns]')])
    def test_table_convert_date_as_object(self, coerce_to_ns, expected_type):
        df = pd.DataFrame({'date': [date(2000, 1, 1), None, date(1970, 1, 1), date(2040, 2, 26)]})
        table = pa.Table.from_pandas(df, preserve_index=False)
        df_datetime = table.to_pandas(date_as_object=False, coerce_temporal_nanoseconds=coerce_to_ns)
        df_object = table.to_pandas()
        tm.assert_frame_equal(df.astype(expected_type), df_datetime, check_dtype=True)
        tm.assert_frame_equal(df, df_object, check_dtype=True)

    @pytest.mark.parametrize('arrow_type', [pa.date32(), pa.date64(), pa.timestamp('s'), pa.timestamp('ms'), pa.timestamp('us'), pa.timestamp('ns'), pa.timestamp('s', 'UTC'), pa.timestamp('ms', 'UTC'), pa.timestamp('us', 'UTC'), pa.timestamp('ns', 'UTC')])
    def test_array_coerce_temporal_nanoseconds(self, arrow_type):
        data = [date(2000, 1, 1), datetime(2001, 1, 1)]
        expected = pd.Series(data)
        arr = pa.array(data).cast(arrow_type)
        result = arr.to_pandas(coerce_temporal_nanoseconds=True, date_as_object=False)
        expected_tz = None
        if hasattr(arrow_type, 'tz') and arrow_type.tz is not None:
            expected_tz = 'UTC'
        expected_type = pa.timestamp('ns', expected_tz).to_pandas_dtype()
        tm.assert_series_equal(result, expected.astype(expected_type))

    @pytest.mark.parametrize('arrow_type', [pa.date32(), pa.date64(), pa.timestamp('s'), pa.timestamp('ms'), pa.timestamp('us'), pa.timestamp('ns'), pa.timestamp('s', 'UTC'), pa.timestamp('ms', 'UTC'), pa.timestamp('us', 'UTC'), pa.timestamp('ns', 'UTC')])
    def test_table_coerce_temporal_nanoseconds(self, arrow_type):
        data = [date(2000, 1, 1), datetime(2001, 1, 1)]
        schema = pa.schema([pa.field('date', arrow_type)])
        expected_df = pd.DataFrame({'date': data})
        table = pa.table([pa.array(data)], schema=schema)
        result_df = table.to_pandas(coerce_temporal_nanoseconds=True, date_as_object=False)
        expected_tz = None
        if hasattr(arrow_type, 'tz') and arrow_type.tz is not None:
            expected_tz = 'UTC'
        expected_type = pa.timestamp('ns', expected_tz).to_pandas_dtype()
        tm.assert_frame_equal(result_df, expected_df.astype(expected_type))

    def test_date_infer(self):
        df = pd.DataFrame({'date': [date(2000, 1, 1), None, date(1970, 1, 1), date(2040, 2, 26)]})
        table = pa.Table.from_pandas(df, preserve_index=False)
        field = pa.field('date', pa.date32())
        expected_schema = pa.schema([field], metadata=table.schema.metadata)
        assert table.schema.equals(expected_schema)
        result = table.to_pandas()
        tm.assert_frame_equal(result, df)

    def test_date_mask(self):
        arr = np.array([date(2017, 4, 3), date(2017, 4, 4)], dtype='datetime64[D]')
        mask = [True, False]
        result = pa.array(arr, mask=np.array(mask))
        expected = np.array([None, date(2017, 4, 4)], dtype='datetime64[D]')
        expected = pa.array(expected, from_pandas=True)
        assert expected.equals(result)

    def test_date_objects_typed(self):
        arr = np.array([date(2017, 4, 3), None, date(2017, 4, 4), date(2017, 4, 5)], dtype=object)
        arr_i4 = np.array([17259, -1, 17260, 17261], dtype='int32')
        arr_i8 = arr_i4.astype('int64') * 86400000
        mask = np.array([False, True, False, False])
        t32 = pa.date32()
        t64 = pa.date64()
        a32 = pa.array(arr, type=t32)
        a64 = pa.array(arr, type=t64)
        a32_expected = pa.array(arr_i4, mask=mask, type=t32)
        a64_expected = pa.array(arr_i8, mask=mask, type=t64)
        assert a32.equals(a32_expected)
        assert a64.equals(a64_expected)
        colnames = ['date32', 'date64']
        table = pa.Table.from_arrays([a32, a64], colnames)
        ex_values = np.array(['2017-04-03', '2017-04-04', '2017-04-04', '2017-04-05'], dtype='datetime64[D]')
        ex_values[1] = pd.NaT.value
        ex_datetime64ms = ex_values.astype('datetime64[ms]')
        expected_pandas = pd.DataFrame({'date32': ex_datetime64ms, 'date64': ex_datetime64ms}, columns=colnames)
        table_pandas = table.to_pandas(date_as_object=False)
        tm.assert_frame_equal(table_pandas, expected_pandas)
        table_pandas_objects = table.to_pandas()
        ex_objects = ex_values.astype('object')
        expected_pandas_objects = pd.DataFrame({'date32': ex_objects, 'date64': ex_objects}, columns=colnames)
        tm.assert_frame_equal(table_pandas_objects, expected_pandas_objects)

    def test_pandas_null_values(self):
        pd_NA = getattr(pd, 'NA', None)
        values = np.array([datetime(2000, 1, 1), pd.NaT, pd_NA], dtype=object)
        values_with_none = np.array([datetime(2000, 1, 1), None, None], dtype=object)
        result = pa.array(values, from_pandas=True)
        expected = pa.array(values_with_none, from_pandas=True)
        assert result.equals(expected)
        assert result.null_count == 2
        assert pa.array([pd.NaT], from_pandas=True).type == pa.null()
        assert pa.array([pd_NA], from_pandas=True).type == pa.null()

    def test_dates_from_integers(self):
        t1 = pa.date32()
        t2 = pa.date64()
        arr = np.array([17259, 17260, 17261], dtype='int32')
        arr2 = arr.astype('int64') * 86400000
        a1 = pa.array(arr, type=t1)
        a2 = pa.array(arr2, type=t2)
        expected = date(2017, 4, 3)
        assert a1[0].as_py() == expected
        assert a2[0].as_py() == expected

    def test_pytime_from_pandas(self):
        pytimes = [time(1, 2, 3, 1356), time(4, 5, 6, 1356)]
        t1 = pa.time64('us')
        aobjs = np.array(pytimes + [None], dtype=object)
        parr = pa.array(aobjs)
        assert parr.type == t1
        assert parr[0].as_py() == pytimes[0]
        assert parr[1].as_py() == pytimes[1]
        assert parr[2].as_py() is None
        df = pd.DataFrame({'times': aobjs})
        batch = pa.RecordBatch.from_pandas(df)
        assert batch[0].equals(parr)
        arr = np.array([_pytime_to_micros(v) for v in pytimes], dtype='int64')
        a1 = pa.array(arr, type=pa.time64('us'))
        assert a1[0].as_py() == pytimes[0]
        a2 = pa.array(arr * 1000, type=pa.time64('ns'))
        assert a2[0].as_py() == pytimes[0]
        a3 = pa.array((arr / 1000).astype('i4'), type=pa.time32('ms'))
        assert a3[0].as_py() == pytimes[0].replace(microsecond=1000)
        a4 = pa.array((arr / 1000000).astype('i4'), type=pa.time32('s'))
        assert a4[0].as_py() == pytimes[0].replace(microsecond=0)

    def test_arrow_time_to_pandas(self):
        pytimes = [time(1, 2, 3, 1356), time(4, 5, 6, 1356), time(0, 0, 0)]
        expected = np.array(pytimes[:2] + [None])
        expected_ms = np.array([x.replace(microsecond=1000) for x in pytimes[:2]] + [None])
        expected_s = np.array([x.replace(microsecond=0) for x in pytimes[:2]] + [None])
        arr = np.array([_pytime_to_micros(v) for v in pytimes], dtype='int64')
        arr = np.array([_pytime_to_micros(v) for v in pytimes], dtype='int64')
        null_mask = np.array([False, False, True], dtype=bool)
        a1 = pa.array(arr, mask=null_mask, type=pa.time64('us'))
        a2 = pa.array(arr * 1000, mask=null_mask, type=pa.time64('ns'))
        a3 = pa.array((arr / 1000).astype('i4'), mask=null_mask, type=pa.time32('ms'))
        a4 = pa.array((arr / 1000000).astype('i4'), mask=null_mask, type=pa.time32('s'))
        names = ['time64[us]', 'time64[ns]', 'time32[ms]', 'time32[s]']
        batch = pa.RecordBatch.from_arrays([a1, a2, a3, a4], names)
        for arr, expected_values in [(a1, expected), (a2, expected), (a3, expected_ms), (a4, expected_s)]:
            result_pandas = arr.to_pandas()
            assert (result_pandas.values == expected_values).all()
        df = batch.to_pandas()
        expected_df = pd.DataFrame({'time64[us]': expected, 'time64[ns]': expected, 'time32[ms]': expected_ms, 'time32[s]': expected_s}, columns=names)
        tm.assert_frame_equal(df, expected_df)

    def test_numpy_datetime64_columns(self):
        datetime64_ns = np.array(['2007-07-13T01:23:34.123456789', None, '2006-01-13T12:34:56.432539784', '2010-08-13T05:46:57.437699912'], dtype='datetime64[ns]')
        _check_array_from_pandas_roundtrip(datetime64_ns)
        datetime64_us = np.array(['2007-07-13T01:23:34.123456', None, '2006-01-13T12:34:56.432539', '2010-08-13T05:46:57.437699'], dtype='datetime64[us]')
        _check_array_from_pandas_roundtrip(datetime64_us)
        datetime64_ms = np.array(['2007-07-13T01:23:34.123', None, '2006-01-13T12:34:56.432', '2010-08-13T05:46:57.437'], dtype='datetime64[ms]')
        _check_array_from_pandas_roundtrip(datetime64_ms)
        datetime64_s = np.array(['2007-07-13T01:23:34', None, '2006-01-13T12:34:56', '2010-08-13T05:46:57'], dtype='datetime64[s]')
        _check_array_from_pandas_roundtrip(datetime64_s)

    def test_timestamp_to_pandas_coerces_to_ns(self):
        if Version(pd.__version__) >= Version('2.0.0'):
            pytest.skip('pandas >= 2.0 supports non-nanosecond datetime64')
        arr = pa.array([1, 2, 3], pa.timestamp('ms'))
        expected = pd.Series(pd.to_datetime([1, 2, 3], unit='ms'))
        s = arr.to_pandas()
        tm.assert_series_equal(s, expected)
        arr = pa.chunked_array([arr])
        s = arr.to_pandas()
        tm.assert_series_equal(s, expected)

    def test_timestamp_to_pandas_out_of_bounds(self):
        for unit in ['s', 'ms', 'us']:
            for tz in [None, 'America/New_York']:
                arr = pa.array([datetime(1, 1, 1)], pa.timestamp(unit, tz=tz))
                table = pa.table({'a': arr})
                msg = 'would result in out of bounds timestamp'
                with pytest.raises(ValueError, match=msg):
                    arr.to_pandas(coerce_temporal_nanoseconds=True)
                with pytest.raises(ValueError, match=msg):
                    table.to_pandas(coerce_temporal_nanoseconds=True)
                with pytest.raises(ValueError, match=msg):
                    table.column('a').to_pandas(coerce_temporal_nanoseconds=True)
                arr.to_pandas(safe=False, coerce_temporal_nanoseconds=True)
                table.to_pandas(safe=False, coerce_temporal_nanoseconds=True)
                table.column('a').to_pandas(safe=False, coerce_temporal_nanoseconds=True)

    def test_timestamp_to_pandas_empty_chunked(self):
        table = pa.table({'a': pa.chunked_array([], type=pa.timestamp('us'))})
        result = table.to_pandas()
        expected = pd.DataFrame({'a': pd.Series([], dtype='datetime64[us]')})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', [pa.date32(), pa.date64()])
    def test_numpy_datetime64_day_unit(self, dtype):
        datetime64_d = np.array(['2007-07-13', None, '2006-01-15', '2010-08-19'], dtype='datetime64[D]')
        _check_array_from_pandas_roundtrip(datetime64_d, type=dtype)

    def test_array_from_pandas_date_with_mask(self):
        m = np.array([True, False, True])
        data = pd.Series([date(1990, 1, 1), date(1991, 1, 1), date(1992, 1, 1)])
        result = pa.Array.from_pandas(data, mask=m)
        expected = pd.Series([None, date(1991, 1, 1), None])
        assert pa.Array.from_pandas(expected).equals(result)

    @pytest.mark.skipif(Version('1.16.0') <= Version(np.__version__) < Version('1.16.1'), reason='Until numpy/numpy#12745 is resolved')
    def test_fixed_offset_timezone(self):
        df = pd.DataFrame({'a': [pd.Timestamp('2012-11-11 00:00:00+01:00'), pd.NaT]})
        _check_pandas_roundtrip(df, check_dtype=False)

    @pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
    def test_timedeltas_no_nulls(self, unit):
        if Version(pd.__version__) < Version('2.0.0'):
            unit = 'ns'
        df = pd.DataFrame({'timedelta64': np.array([0, 3600000000000, 7200000000000], dtype=f'timedelta64[{unit}]')})
        field = pa.field('timedelta64', pa.duration(unit))
        schema = pa.schema([field])
        _check_pandas_roundtrip(df, expected_schema=schema)

    @pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
    def test_timedeltas_nulls(self, unit):
        if Version(pd.__version__) < Version('2.0.0'):
            unit = 'ns'
        df = pd.DataFrame({'timedelta64': np.array([0, None, 7200000000000], dtype=f'timedelta64[{unit}]')})
        field = pa.field('timedelta64', pa.duration(unit))
        schema = pa.schema([field])
        _check_pandas_roundtrip(df, expected_schema=schema)

    def test_month_day_nano_interval(self):
        from pandas.tseries.offsets import DateOffset
        df = pd.DataFrame({'date_offset': [None, DateOffset(days=3600, months=3600, microseconds=3, nanoseconds=600)]})
        schema = pa.schema([('date_offset', pa.month_day_nano_interval())])
        _check_pandas_roundtrip(df, expected_schema=schema)