import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
from pandas._config import using_copy_on_write
from pandas._config.config import _get_option
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import (
class TestParquetPyArrow(Base):

    def test_basic(self, pa, df_full):
        df = df_full
        dti = pd.date_range('20130101', periods=3, tz='Europe/Brussels')
        dti = dti._with_freq(None)
        df['datetime_tz'] = dti
        df['bool_with_none'] = [True, None, True]
        check_round_trip(df, pa)

    def test_basic_subset_columns(self, pa, df_full):
        df = df_full
        df['datetime_tz'] = pd.date_range('20130101', periods=3, tz='Europe/Brussels')
        check_round_trip(df, pa, expected=df[['string', 'int']], read_kwargs={'columns': ['string', 'int']})

    def test_to_bytes_without_path_or_buf_provided(self, pa, df_full):
        buf_bytes = df_full.to_parquet(engine=pa)
        assert isinstance(buf_bytes, bytes)
        buf_stream = BytesIO(buf_bytes)
        res = read_parquet(buf_stream)
        expected = df_full.copy()
        expected.loc[1, 'string_with_nan'] = None
        tm.assert_frame_equal(res, expected)

    def test_duplicate_columns(self, pa):
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list('aaa')).copy()
        self.check_error_on_write(df, pa, ValueError, 'Duplicate column names found')

    def test_timedelta(self, pa):
        df = pd.DataFrame({'a': pd.timedelta_range('1 day', periods=3)})
        check_round_trip(df, pa)

    def test_unsupported(self, pa):
        df = pd.DataFrame({'a': ['a', 1, 2.0]})
        self.check_external_error_on_write(df, pa, pyarrow.ArrowException)

    def test_unsupported_float16(self, pa):
        data = np.arange(2, 10, dtype=np.float16)
        df = pd.DataFrame(data=data, columns=['fp16'])
        if pa_version_under15p0:
            self.check_external_error_on_write(df, pa, pyarrow.ArrowException)
        else:
            check_round_trip(df, pa)

    @pytest.mark.xfail(is_platform_windows(), reason='PyArrow does not cleanup of partial files dumps when unsupported dtypes are passed to_parquet function in windows')
    @pytest.mark.skipif(not pa_version_under15p0, reason='float16 works on 15')
    @pytest.mark.parametrize('path_type', [str, pathlib.Path])
    def test_unsupported_float16_cleanup(self, pa, path_type):
        data = np.arange(2, 10, dtype=np.float16)
        df = pd.DataFrame(data=data, columns=['fp16'])
        with tm.ensure_clean() as path_str:
            path = path_type(path_str)
            with tm.external_error_raised(pyarrow.ArrowException):
                df.to_parquet(path=path, engine=pa)
            assert not os.path.isfile(path)

    def test_categorical(self, pa):
        df = pd.DataFrame()
        df['a'] = pd.Categorical(list('abcdef'))
        df['b'] = pd.Categorical(['bar', 'foo', 'foo', 'bar', None, 'bar'], dtype=pd.CategoricalDtype(['foo', 'bar', 'baz']))
        df['c'] = pd.Categorical(['a', 'b', 'c', 'a', 'c', 'b'], categories=['b', 'c', 'd'], ordered=True)
        check_round_trip(df, pa)

    @pytest.mark.single_cpu
    def test_s3_roundtrip_explicit_fs(self, df_compat, s3_public_bucket, pa, s3so):
        s3fs = pytest.importorskip('s3fs')
        s3 = s3fs.S3FileSystem(**s3so)
        kw = {'filesystem': s3}
        check_round_trip(df_compat, pa, path=f'{s3_public_bucket.name}/pyarrow.parquet', read_kwargs=kw, write_kwargs=kw)

    @pytest.mark.single_cpu
    def test_s3_roundtrip(self, df_compat, s3_public_bucket, pa, s3so):
        s3so = {'storage_options': s3so}
        check_round_trip(df_compat, pa, path=f's3://{s3_public_bucket.name}/pyarrow.parquet', read_kwargs=s3so, write_kwargs=s3so)

    @pytest.mark.single_cpu
    @pytest.mark.parametrize('partition_col', [['A'], []])
    def test_s3_roundtrip_for_dir(self, df_compat, s3_public_bucket, pa, partition_col, s3so):
        pytest.importorskip('s3fs')
        expected_df = df_compat.copy()
        if partition_col:
            expected_df = expected_df.astype(dict.fromkeys(partition_col, np.int32))
            partition_col_type = 'category'
            expected_df[partition_col] = expected_df[partition_col].astype(partition_col_type)
        check_round_trip(df_compat, pa, expected=expected_df, path=f's3://{s3_public_bucket.name}/parquet_dir', read_kwargs={'storage_options': s3so}, write_kwargs={'partition_cols': partition_col, 'compression': None, 'storage_options': s3so}, check_like=True, repeat=1)

    def test_read_file_like_obj_support(self, df_compat):
        pytest.importorskip('pyarrow')
        buffer = BytesIO()
        df_compat.to_parquet(buffer)
        df_from_buf = read_parquet(buffer)
        tm.assert_frame_equal(df_compat, df_from_buf)

    def test_expand_user(self, df_compat, monkeypatch):
        pytest.importorskip('pyarrow')
        monkeypatch.setenv('HOME', 'TestingUser')
        monkeypatch.setenv('USERPROFILE', 'TestingUser')
        with pytest.raises(OSError, match='.*TestingUser.*'):
            read_parquet('~/file.parquet')
        with pytest.raises(OSError, match='.*TestingUser.*'):
            df_compat.to_parquet('~/file.parquet')

    def test_partition_cols_supported(self, tmp_path, pa, df_full):
        partition_cols = ['bool', 'int']
        df = df_full
        df.to_parquet(tmp_path, partition_cols=partition_cols, compression=None)
        check_partition_names(tmp_path, partition_cols)
        assert read_parquet(tmp_path).shape == df.shape

    def test_partition_cols_string(self, tmp_path, pa, df_full):
        partition_cols = 'bool'
        partition_cols_list = [partition_cols]
        df = df_full
        df.to_parquet(tmp_path, partition_cols=partition_cols, compression=None)
        check_partition_names(tmp_path, partition_cols_list)
        assert read_parquet(tmp_path).shape == df.shape

    @pytest.mark.parametrize('path_type', [str, lambda x: x], ids=['string', 'pathlib.Path'])
    def test_partition_cols_pathlib(self, tmp_path, pa, df_compat, path_type):
        partition_cols = 'B'
        partition_cols_list = [partition_cols]
        df = df_compat
        path = path_type(tmp_path)
        df.to_parquet(path, partition_cols=partition_cols_list)
        assert read_parquet(path).shape == df.shape

    def test_empty_dataframe(self, pa):
        df = pd.DataFrame(index=[], columns=[])
        check_round_trip(df, pa)

    def test_write_with_schema(self, pa):
        import pyarrow
        df = pd.DataFrame({'x': [0, 1]})
        schema = pyarrow.schema([pyarrow.field('x', type=pyarrow.bool_())])
        out_df = df.astype(bool)
        check_round_trip(df, pa, write_kwargs={'schema': schema}, expected=out_df)

    def test_additional_extension_arrays(self, pa):
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({'a': pd.Series([1, 2, 3], dtype='Int64'), 'b': pd.Series([1, 2, 3], dtype='UInt32'), 'c': pd.Series(['a', None, 'c'], dtype='string')})
        check_round_trip(df, pa)
        df = pd.DataFrame({'a': pd.Series([1, 2, 3, None], dtype='Int64')})
        check_round_trip(df, pa)

    def test_pyarrow_backed_string_array(self, pa, string_storage):
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({'a': pd.Series(['a', None, 'c'], dtype='string[pyarrow]')})
        with pd.option_context('string_storage', string_storage):
            check_round_trip(df, pa, expected=df.astype(f'string[{string_storage}]'))

    def test_additional_extension_types(self, pa):
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({'c': pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)]), 'd': pd.period_range('2012-01-01', periods=3, freq='D'), 'e': pd.IntervalIndex.from_breaks(pd.date_range('2012-01-01', periods=4, freq='D'))})
        check_round_trip(df, pa)

    def test_timestamp_nanoseconds(self, pa):
        ver = '2.6'
        df = pd.DataFrame({'a': pd.date_range('2017-01-01', freq='1ns', periods=10)})
        check_round_trip(df, pa, write_kwargs={'version': ver})

    def test_timezone_aware_index(self, request, pa, timezone_aware_date_list):
        if timezone_aware_date_list.tzinfo != datetime.timezone.utc:
            request.applymarker(pytest.mark.xfail(reason='temporary skip this test until it is properly resolved: https://github.com/pandas-dev/pandas/issues/37286'))
        idx = 5 * [timezone_aware_date_list]
        df = pd.DataFrame(index=idx, data={'index_as_col': idx})
        check_round_trip(df, pa, check_dtype=False)

    def test_filter_row_groups(self, pa):
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({'a': list(range(3))})
        with tm.ensure_clean() as path:
            df.to_parquet(path, engine=pa)
            result = read_parquet(path, pa, filters=[('a', '==', 0)])
        assert len(result) == 1

    def test_read_parquet_manager(self, pa, using_array_manager):
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['A', 'B', 'C'])
        with tm.ensure_clean() as path:
            df.to_parquet(path, engine=pa)
            result = read_parquet(path, pa)
        if using_array_manager:
            assert isinstance(result._mgr, pd.core.internals.ArrayManager)
        else:
            assert isinstance(result._mgr, pd.core.internals.BlockManager)

    def test_read_dtype_backend_pyarrow_config(self, pa, df_full):
        import pyarrow
        df = df_full
        dti = pd.date_range('20130101', periods=3, tz='Europe/Brussels')
        dti = dti._with_freq(None)
        df['datetime_tz'] = dti
        df['bool_with_none'] = [True, None, True]
        pa_table = pyarrow.Table.from_pandas(df)
        expected = pa_table.to_pandas(types_mapper=pd.ArrowDtype)
        if pa_version_under13p0:
            expected['datetime'] = expected['datetime'].astype('timestamp[us][pyarrow]')
            expected['datetime_with_nat'] = expected['datetime_with_nat'].astype('timestamp[us][pyarrow]')
            expected['datetime_tz'] = expected['datetime_tz'].astype(pd.ArrowDtype(pyarrow.timestamp(unit='us', tz='Europe/Brussels')))
        check_round_trip(df, engine=pa, read_kwargs={'dtype_backend': 'pyarrow'}, expected=expected)

    def test_read_dtype_backend_pyarrow_config_index(self, pa):
        df = pd.DataFrame({'a': [1, 2]}, index=pd.Index([3, 4], name='test'), dtype='int64[pyarrow]')
        expected = df.copy()
        import pyarrow
        if Version(pyarrow.__version__) > Version('11.0.0'):
            expected.index = expected.index.astype('int64[pyarrow]')
        check_round_trip(df, engine=pa, read_kwargs={'dtype_backend': 'pyarrow'}, expected=expected)

    def test_columns_dtypes_not_invalid(self, pa):
        df = pd.DataFrame({'string': list('abc'), 'int': list(range(1, 4))})
        df.columns = [0, 1]
        check_round_trip(df, pa)
        df.columns = [b'foo', b'bar']
        with pytest.raises(NotImplementedError, match='|S3'):
            check_round_trip(df, pa)
        df.columns = [datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2011, 1, 1, 1, 1)]
        check_round_trip(df, pa)

    def test_empty_columns(self, pa):
        df = pd.DataFrame(index=pd.Index(['a', 'b', 'c'], name='custom name'))
        check_round_trip(df, pa)

    def test_df_attrs_persistence(self, tmp_path, pa):
        path = tmp_path / 'test_df_metadata.p'
        df = pd.DataFrame(data={1: [1]})
        df.attrs = {'test_attribute': 1}
        df.to_parquet(path, engine=pa)
        new_df = read_parquet(path, engine=pa)
        assert new_df.attrs == df.attrs

    def test_string_inference(self, tmp_path, pa):
        path = tmp_path / 'test_string_inference.p'
        df = pd.DataFrame(data={'a': ['x', 'y']}, index=['a', 'b'])
        df.to_parquet(path, engine='pyarrow')
        with pd.option_context('future.infer_string', True):
            result = read_parquet(path, engine='pyarrow')
        expected = pd.DataFrame(data={'a': ['x', 'y']}, dtype='string[pyarrow_numpy]', index=pd.Index(['a', 'b'], dtype='string[pyarrow_numpy]'))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.skipif(pa_version_under11p0, reason='not supported before 11.0')
    def test_roundtrip_decimal(self, tmp_path, pa):
        import pyarrow as pa
        path = tmp_path / 'decimal.p'
        df = pd.DataFrame({'a': [Decimal('123.00')]}, dtype='string[pyarrow]')
        df.to_parquet(path, schema=pa.schema([('a', pa.decimal128(5))]))
        result = read_parquet(path)
        expected = pd.DataFrame({'a': ['123']}, dtype='string[python]')
        tm.assert_frame_equal(result, expected)

    def test_infer_string_large_string_type(self, tmp_path, pa):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = tmp_path / 'large_string.p'
        table = pa.table({'a': pa.array([None, 'b', 'c'], pa.large_string())})
        pq.write_table(table, path)
        with pd.option_context('future.infer_string', True):
            result = read_parquet(path)
        expected = pd.DataFrame(data={'a': [None, 'b', 'c']}, dtype='string[pyarrow_numpy]', columns=pd.Index(['a'], dtype='string[pyarrow_numpy]'))
        tm.assert_frame_equal(result, expected)