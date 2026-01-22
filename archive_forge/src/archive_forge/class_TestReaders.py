from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
class TestReaders:

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine, datapath, monkeypatch):
        """
        Change directory and set engine for read_excel calls.
        """
        func = partial(pd.read_excel, engine=engine)
        monkeypatch.chdir(datapath('io', 'data', 'excel'))
        monkeypatch.setattr(pd, 'read_excel', func)

    def test_engine_used(self, read_ext, engine, monkeypatch):

        def parser(self, *args, **kwargs):
            return self.engine
        monkeypatch.setattr(pd.ExcelFile, 'parse', parser)
        expected_defaults = {'xlsx': 'openpyxl', 'xlsm': 'openpyxl', 'xlsb': 'pyxlsb', 'xls': 'xlrd', 'ods': 'odf'}
        with open('test1' + read_ext, 'rb') as f:
            result = pd.read_excel(f)
        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_engine_kwargs(self, read_ext, engine):
        expected_defaults = {'xlsx': {'foo': 'abcd'}, 'xlsm': {'foo': 123}, 'xlsb': {'foo': 'True'}, 'xls': {'foo': True}, 'ods': {'foo': 'abcd'}}
        if engine in {'xlrd', 'pyxlsb'}:
            msg = re.escape("open_workbook() got an unexpected keyword argument 'foo'")
        elif engine == 'odf':
            msg = re.escape("load() got an unexpected keyword argument 'foo'")
        else:
            msg = re.escape("load_workbook() got an unexpected keyword argument 'foo'")
        if engine is not None:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, engine_kwargs=expected_defaults[read_ext[1:]])

    def test_usecols_int(self, read_ext):
        msg = 'Passing an integer for `usecols`'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols=3)
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols=3)

    def test_usecols_list(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)
        expected = df_ref[['B', 'C']]
        adjust_expected(expected, read_ext, engine)
        df1 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols=[0, 2, 3])
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols=[0, 2, 3])
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_usecols_str(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)
        expected = df_ref[['A', 'B', 'C']]
        adjust_expected(expected, read_ext, engine)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A:D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols='A:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        expected = df_ref[['B', 'C']]
        adjust_expected(expected, read_ext, engine)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A,C,D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols='A,C,D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A,C:D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols='A,C:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize('usecols', [[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0]])
    def test_usecols_diff_positional_int_columns_order(self, request, engine, read_ext, usecols, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)
        expected = df_ref[['A', 'C']]
        adjust_expected(expected, read_ext, engine)
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols=usecols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('usecols', [['B', 'D'], ['D', 'B']])
    def test_usecols_diff_positional_str_columns_order(self, read_ext, usecols, df_ref):
        expected = df_ref[['B', 'D']]
        expected.index = range(len(expected))
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', usecols=usecols)
        tm.assert_frame_equal(result, expected)

    def test_read_excel_without_slicing(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)
        expected = df_ref
        adjust_expected(expected, read_ext, engine)
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0)
        tm.assert_frame_equal(result, expected)

    def test_usecols_excel_range_str(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)
        expected = df_ref[['C', 'D']]
        adjust_expected(expected, read_ext, engine)
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A,D:E')
        tm.assert_frame_equal(result, expected)

    def test_usecols_excel_range_str_invalid(self, read_ext):
        msg = 'Invalid column name: E1'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1', usecols='D:E1')

    def test_index_col_label_error(self, read_ext):
        msg = 'list indices must be integers.*, not str'
        with pytest.raises(TypeError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=['A'], usecols=['A', 'C'])

    def test_index_col_str(self, read_ext):
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet3', index_col='A')
        expected = DataFrame(columns=['B', 'C', 'D', 'E', 'F'], index=Index([], name='A'))
        tm.assert_frame_equal(result, expected)

    def test_index_col_empty(self, read_ext):
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet3', index_col=['A', 'B', 'C'])
        expected = DataFrame(columns=['D', 'E', 'F'], index=MultiIndex(levels=[[]] * 3, codes=[[]] * 3, names=['A', 'B', 'C']))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index_col', [None, 2])
    def test_index_col_with_unnamed(self, read_ext, index_col):
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet4', index_col=index_col)
        expected = DataFrame([['i1', 'a', 'x'], ['i2', 'b', 'y']], columns=['Unnamed: 0', 'col1', 'col2'])
        if index_col:
            expected = expected.set_index(expected.columns[index_col])
        tm.assert_frame_equal(result, expected)

    def test_usecols_pass_non_existent_column(self, read_ext):
        msg = "Usecols do not match columns, columns expected but not found: \\['E'\\]"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, usecols=['E'])

    def test_usecols_wrong_type(self, read_ext):
        msg = "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, usecols=['E1', 0])

    def test_excel_stop_iterator(self, read_ext):
        parsed = pd.read_excel('test2' + read_ext, sheet_name='Sheet1')
        expected = DataFrame([['aaaa', 'bbbbb']], columns=['Test', 'Test1'])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_cell_error_na(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)
        if engine == 'calamine' and read_ext == '.ods':
            request.applymarker(pytest.mark.xfail(reason="Calamine can't extract error from ods files"))
        parsed = pd.read_excel('test3' + read_ext, sheet_name='Sheet1')
        expected = DataFrame([[np.nan]], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_table(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)
        expected = df_ref
        adjust_expected(expected, read_ext, engine)
        df1 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def test_reader_special_dtypes(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)
        unit = get_exp_unit(read_ext, engine)
        expected = DataFrame.from_dict({'IntCol': [1, 2, -3, 4, 0], 'FloatCol': [1.25, 2.25, 1.83, 1.92, 5e-10], 'BoolCol': [True, False, True, True, False], 'StrCol': [1, 2, 3, 4, 5], 'Str2Col': ['a', 3, 'c', 'd', 'e'], 'DateCol': Index([datetime(2013, 10, 30), datetime(2013, 10, 31), datetime(1905, 1, 1), datetime(2013, 12, 14), datetime(2015, 3, 14)], dtype=f'M8[{unit}]')})
        basename = 'test_types'
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, expected)
        float_expected = expected.copy()
        float_expected.loc[float_expected.index[1], 'Str2Col'] = 3.0
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, float_expected)
        for icol, name in enumerate(expected.columns):
            actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1', index_col=icol)
            exp = expected.set_index(name)
            tm.assert_frame_equal(actual, exp)
        expected['StrCol'] = expected['StrCol'].apply(str)
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1', converters={'StrCol': str})
        tm.assert_frame_equal(actual, expected)

    def test_reader_converters(self, read_ext):
        basename = 'test_converters'
        expected = DataFrame.from_dict({'IntCol': [1, 2, -3, -1000, 0], 'FloatCol': [12.5, np.nan, 18.3, 19.2, 5e-09], 'BoolCol': ['Found', 'Found', 'Found', 'Not found', 'Found'], 'StrCol': ['1', np.nan, '3', '4', '5']})
        converters = {'IntCol': lambda x: int(x) if x != '' else -1000, 'FloatCol': lambda x: 10 * x if x else np.nan, 2: lambda x: 'Found' if x != '' else 'Not found', 3: lambda x: str(x) if x else ''}
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1', converters=converters)
        tm.assert_frame_equal(actual, expected)

    def test_reader_dtype(self, read_ext):
        basename = 'testdtype'
        actual = pd.read_excel(basename + read_ext)
        expected = DataFrame({'a': [1, 2, 3, 4], 'b': [2.5, 3.5, 4.5, 5.5], 'c': [1, 2, 3, 4], 'd': [1.0, 2.0, np.nan, 4.0]})
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(basename + read_ext, dtype={'a': 'float64', 'b': 'float32', 'c': str})
        expected['a'] = expected['a'].astype('float64')
        expected['b'] = expected['b'].astype('float32')
        expected['c'] = Series(['001', '002', '003', '004'], dtype=object)
        tm.assert_frame_equal(actual, expected)
        msg = 'Unable to convert column d to type int64'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(basename + read_ext, dtype={'d': 'int64'})

    @pytest.mark.parametrize('dtype,expected', [(None, DataFrame({'a': [1, 2, 3, 4], 'b': [2.5, 3.5, 4.5, 5.5], 'c': [1, 2, 3, 4], 'd': [1.0, 2.0, np.nan, 4.0]})), ({'a': 'float64', 'b': 'float32', 'c': str, 'd': str}, DataFrame({'a': Series([1, 2, 3, 4], dtype='float64'), 'b': Series([2.5, 3.5, 4.5, 5.5], dtype='float32'), 'c': Series(['001', '002', '003', '004'], dtype=object), 'd': Series(['1', '2', np.nan, '4'], dtype=object)}))])
    def test_reader_dtype_str(self, read_ext, dtype, expected):
        basename = 'testdtype'
        actual = pd.read_excel(basename + read_ext, dtype=dtype)
        tm.assert_frame_equal(actual, expected)

    def test_dtype_backend(self, read_ext, dtype_backend, engine):
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({'a': Series([1, 3], dtype='Int64'), 'b': Series([2.5, 4.5], dtype='Float64'), 'c': Series([True, False], dtype='boolean'), 'd': Series(['a', 'b'], dtype='string'), 'e': Series([pd.NA, 6], dtype='Int64'), 'f': Series([pd.NA, 7.5], dtype='Float64'), 'g': Series([pd.NA, True], dtype='boolean'), 'h': Series([pd.NA, 'a'], dtype='string'), 'i': Series([pd.Timestamp('2019-12-31')] * 2), 'j': Series([pd.NA, pd.NA], dtype='Int64')})
        with tm.ensure_clean(read_ext) as file_path:
            df.to_excel(file_path, sheet_name='test', index=False)
            result = pd.read_excel(file_path, sheet_name='test', dtype_backend=dtype_backend)
        if dtype_backend == 'pyarrow':
            import pyarrow as pa
            from pandas.arrays import ArrowExtensionArray
            expected = DataFrame({col: ArrowExtensionArray(pa.array(df[col], from_pandas=True)) for col in df.columns})
            expected['i'] = ArrowExtensionArray(expected['i'].array._pa_array.cast(pa.timestamp(unit='us')))
            expected['j'] = ArrowExtensionArray(pa.array([None, None]))
        else:
            expected = df
            unit = get_exp_unit(read_ext, engine)
            expected['i'] = expected['i'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(result, expected)

    def test_dtype_backend_and_dtype(self, read_ext):
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({'a': [np.nan, 1.0], 'b': [2.5, np.nan]})
        with tm.ensure_clean(read_ext) as file_path:
            df.to_excel(file_path, sheet_name='test', index=False)
            result = pd.read_excel(file_path, sheet_name='test', dtype_backend='numpy_nullable', dtype='float64')
        tm.assert_frame_equal(result, df)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason='infer_string takes precedence')
    def test_dtype_backend_string(self, read_ext, string_storage):
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        pa = pytest.importorskip('pyarrow')
        with pd.option_context('mode.string_storage', string_storage):
            df = DataFrame({'a': np.array(['a', 'b'], dtype=np.object_), 'b': np.array(['x', pd.NA], dtype=np.object_)})
            with tm.ensure_clean(read_ext) as file_path:
                df.to_excel(file_path, sheet_name='test', index=False)
                result = pd.read_excel(file_path, sheet_name='test', dtype_backend='numpy_nullable')
            if string_storage == 'python':
                expected = DataFrame({'a': StringArray(np.array(['a', 'b'], dtype=np.object_)), 'b': StringArray(np.array(['x', pd.NA], dtype=np.object_))})
            else:
                expected = DataFrame({'a': ArrowStringArray(pa.array(['a', 'b'])), 'b': ArrowStringArray(pa.array(['x', None]))})
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtypes, exp_value', [({}, 1), ({'a.1': 'int64'}, 1)])
    def test_dtype_mangle_dup_cols(self, read_ext, dtypes, exp_value):
        basename = 'df_mangle_dup_col_dtypes'
        dtype_dict = {'a': object, **dtypes}
        dtype_dict_copy = dtype_dict.copy()
        result = pd.read_excel(basename + read_ext, dtype=dtype_dict)
        expected = DataFrame({'a': Series([1], dtype=object), 'a.1': Series([exp_value], dtype=object if not dtypes else None)})
        assert dtype_dict == dtype_dict_copy, 'dtype dict changed'
        tm.assert_frame_equal(result, expected)

    def test_reader_spaces(self, read_ext):
        basename = 'test_spaces'
        actual = pd.read_excel(basename + read_ext)
        expected = DataFrame({'testcol': ['this is great', '4    spaces', '1 trailing ', ' 1 leading', '2  spaces  multiple  times']})
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize('basename,expected', [('gh-35802', DataFrame({'COLUMN': ['Test (1)']})), ('gh-36122', DataFrame(columns=['got 2nd sa']))])
    def test_read_excel_ods_nested_xml(self, engine, read_ext, basename, expected):
        if engine != 'odf':
            pytest.skip(f'Skipped for engine: {engine}')
        actual = pd.read_excel(basename + read_ext)
        tm.assert_frame_equal(actual, expected)

    def test_reading_all_sheets(self, read_ext):
        basename = 'test_multisheet'
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        expected_keys = ['Charlie', 'Alpha', 'Beta']
        tm.assert_contains_all(expected_keys, dfs.keys())
        assert expected_keys == list(dfs.keys())

    def test_reading_multiple_specific_sheets(self, read_ext):
        basename = 'test_multisheet'
        expected_keys = [2, 'Charlie', 'Charlie']
        dfs = pd.read_excel(basename + read_ext, sheet_name=expected_keys)
        expected_keys = list(set(expected_keys))
        tm.assert_contains_all(expected_keys, dfs.keys())
        assert len(expected_keys) == len(dfs.keys())

    def test_reading_all_sheets_with_blank(self, read_ext):
        basename = 'blank_with_header'
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        expected_keys = ['Sheet1', 'Sheet2', 'Sheet3']
        tm.assert_contains_all(expected_keys, dfs.keys())

    def test_read_excel_blank(self, read_ext):
        actual = pd.read_excel('blank' + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, DataFrame())

    def test_read_excel_blank_with_header(self, read_ext):
        expected = DataFrame(columns=['col_1', 'col_2'])
        actual = pd.read_excel('blank_with_header' + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, expected)

    def test_exception_message_includes_sheet_name(self, read_ext):
        with pytest.raises(ValueError, match=' \\(sheet: Sheet1\\)$'):
            pd.read_excel('blank_with_header' + read_ext, header=[1], sheet_name=None)
        with pytest.raises(ZeroDivisionError, match=' \\(sheet: Sheet1\\)$'):
            pd.read_excel('test1' + read_ext, usecols=lambda x: 1 / 0, sheet_name=None)

    @pytest.mark.filterwarnings('ignore:Cell A4 is marked:UserWarning:openpyxl')
    def test_date_conversion_overflow(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)
        expected = DataFrame([[pd.Timestamp('2016-03-12'), 'Marc Johnson'], [pd.Timestamp('2016-03-16'), 'Jack Black'], [1e+20, 'Timothy Brown']], columns=['DateColWithBigInt', 'StringCol'])
        if engine == 'openpyxl':
            request.applymarker(pytest.mark.xfail(reason='Maybe not supported by openpyxl'))
        if engine is None and read_ext in ('.xlsx', '.xlsm'):
            request.applymarker(pytest.mark.xfail(reason='Defaults to openpyxl, maybe not supported'))
        result = pd.read_excel('testdateoverflow' + read_ext)
        tm.assert_frame_equal(result, expected)

    def test_sheet_name(self, request, read_ext, engine, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)
        filename = 'test1'
        sheet_name = 'Sheet1'
        expected = df_ref
        adjust_expected(expected, read_ext, engine)
        df1 = pd.read_excel(filename + read_ext, sheet_name=sheet_name, index_col=0)
        df2 = pd.read_excel(filename + read_ext, index_col=0, sheet_name=sheet_name)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_excel_read_buffer(self, read_ext):
        pth = 'test1' + read_ext
        expected = pd.read_excel(pth, sheet_name='Sheet1', index_col=0)
        with open(pth, 'rb') as f:
            actual = pd.read_excel(f, sheet_name='Sheet1', index_col=0)
            tm.assert_frame_equal(expected, actual)

    def test_bad_engine_raises(self):
        bad_engine = 'foo'
        with pytest.raises(ValueError, match='Unknown engine: foo'):
            pd.read_excel('', engine=bad_engine)

    @pytest.mark.parametrize('sheet_name', [3, [0, 3], [3, 0], 'Sheet4', ['Sheet1', 'Sheet4'], ['Sheet4', 'Sheet1']])
    def test_bad_sheetname_raises(self, read_ext, sheet_name):
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('blank' + read_ext, sheet_name=sheet_name)

    def test_missing_file_raises(self, read_ext):
        bad_file = f'foo{read_ext}'
        match = '|'.join(['(No such file or directory', '没有那个文件或目录', 'File o directory non esistente)'])
        with pytest.raises(FileNotFoundError, match=match):
            pd.read_excel(bad_file)

    def test_corrupt_bytes_raises(self, engine):
        bad_stream = b'foo'
        if engine is None:
            error = ValueError
            msg = 'Excel file format cannot be determined, you must specify an engine manually.'
        elif engine == 'xlrd':
            from xlrd import XLRDError
            error = XLRDError
            msg = "Unsupported format, or corrupt file: Expected BOF record; found b'foo'"
        elif engine == 'calamine':
            from python_calamine import CalamineError
            error = CalamineError
            msg = 'Cannot detect file format'
        else:
            error = BadZipFile
            msg = 'File is not a zip file'
        with pytest.raises(error, match=msg):
            pd.read_excel(BytesIO(bad_stream))

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_read_from_http_url(self, httpserver, read_ext):
        with open('test1' + read_ext, 'rb') as f:
            httpserver.serve_content(content=f.read())
        url_table = pd.read_excel(httpserver.url)
        local_table = pd.read_excel('test1' + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @td.skip_if_not_us_locale
    @pytest.mark.single_cpu
    def test_read_from_s3_url(self, read_ext, s3_public_bucket, s3so):
        with open('test1' + read_ext, 'rb') as f:
            s3_public_bucket.put_object(Key='test1' + read_ext, Body=f)
        url = f's3://{s3_public_bucket.name}/test1' + read_ext
        url_table = pd.read_excel(url, storage_options=s3so)
        local_table = pd.read_excel('test1' + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.single_cpu
    def test_read_from_s3_object(self, read_ext, s3_public_bucket, s3so):
        with open('test1' + read_ext, 'rb') as f:
            s3_public_bucket.put_object(Key='test1' + read_ext, Body=f)
        import s3fs
        s3 = s3fs.S3FileSystem(**s3so)
        with s3.open(f's3://{s3_public_bucket.name}/test1' + read_ext) as f:
            url_table = pd.read_excel(f)
        local_table = pd.read_excel('test1' + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.slow
    def test_read_from_file_url(self, read_ext, datapath):
        localtable = os.path.join(datapath('io', 'data', 'excel'), 'test1' + read_ext)
        local_table = pd.read_excel(localtable)
        try:
            url_table = pd.read_excel('file://localhost/' + localtable)
        except URLError:
            platform_info = ' '.join(platform.uname()).strip()
            pytest.skip(f'failing on {platform_info}')
        tm.assert_frame_equal(url_table, local_table)

    def test_read_from_pathlib_path(self, read_ext):
        str_path = 'test1' + read_ext
        expected = pd.read_excel(str_path, sheet_name='Sheet1', index_col=0)
        path_obj = Path('test1' + read_ext)
        actual = pd.read_excel(path_obj, sheet_name='Sheet1', index_col=0)
        tm.assert_frame_equal(expected, actual)

    @td.skip_if_no('py.path')
    def test_read_from_py_localpath(self, read_ext):
        from py.path import local as LocalPath
        str_path = os.path.join('test1' + read_ext)
        expected = pd.read_excel(str_path, sheet_name='Sheet1', index_col=0)
        path_obj = LocalPath().join('test1' + read_ext)
        actual = pd.read_excel(path_obj, sheet_name='Sheet1', index_col=0)
        tm.assert_frame_equal(expected, actual)

    def test_close_from_py_localpath(self, read_ext):
        str_path = os.path.join('test1' + read_ext)
        with open(str_path, 'rb') as f:
            x = pd.read_excel(f, sheet_name='Sheet1', index_col=0)
            del x
            f.read()

    def test_reader_seconds(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)
        if engine == 'calamine' and read_ext == '.ods':
            request.applymarker(pytest.mark.xfail(reason='ODS file contains bad datetime (seconds as text)'))
        expected = DataFrame.from_dict({'Time': [time(1, 2, 3), time(2, 45, 56, 100000), time(4, 29, 49, 200000), time(6, 13, 42, 300000), time(7, 57, 35, 400000), time(9, 41, 28, 500000), time(11, 25, 21, 600000), time(13, 9, 14, 700000), time(14, 53, 7, 800000), time(16, 37, 0, 900000), time(18, 20, 54)]})
        actual = pd.read_excel('times_1900' + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel('times_1904' + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_multiindex(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)
        unit = get_exp_unit(read_ext, engine)
        mi = MultiIndex.from_product([['foo', 'bar'], ['a', 'b']])
        mi_file = 'testmultiindex' + read_ext
        expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [2, 3.5, pd.Timestamp('2015-01-02'), False], [3, 4.5, pd.Timestamp('2015-01-03'), False], [4, 5.5, pd.Timestamp('2015-01-04'), True]], columns=mi)
        expected[mi[2]] = expected[mi[2]].astype(f'M8[{unit}]')
        actual = pd.read_excel(mi_file, sheet_name='mi_column', header=[0, 1], index_col=0)
        tm.assert_frame_equal(actual, expected)
        expected.index = mi
        expected.columns = ['a', 'b', 'c', 'd']
        actual = pd.read_excel(mi_file, sheet_name='mi_index', index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)
        expected.columns = mi
        actual = pd.read_excel(mi_file, sheet_name='both', index_col=[0, 1], header=[0, 1])
        tm.assert_frame_equal(actual, expected)
        expected.columns = ['a', 'b', 'c', 'd']
        expected.index = mi.set_names(['ilvl1', 'ilvl2'])
        actual = pd.read_excel(mi_file, sheet_name='mi_index_name', index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)
        expected.index = list(range(4))
        expected.columns = mi.set_names(['c1', 'c2'])
        actual = pd.read_excel(mi_file, sheet_name='mi_column_name', header=[0, 1], index_col=0)
        tm.assert_frame_equal(actual, expected)
        expected.columns = mi.set_levels([1, 2], level=1).set_names(['c1', 'c2'])
        actual = pd.read_excel(mi_file, sheet_name='name_with_int', index_col=0, header=[0, 1])
        tm.assert_frame_equal(actual, expected)
        expected.columns = mi.set_names(['c1', 'c2'])
        expected.index = mi.set_names(['ilvl1', 'ilvl2'])
        actual = pd.read_excel(mi_file, sheet_name='both_name', index_col=[0, 1], header=[0, 1])
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(mi_file, sheet_name='both_name_skiprows', index_col=[0, 1], header=[0, 1], skiprows=2)
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize('sheet_name,idx_lvl2', [('both_name_blank_after_mi_name', [np.nan, 'b', 'a', 'b']), ('both_name_multiple_blanks', [np.nan] * 4)])
    def test_read_excel_multiindex_blank_after_name(self, request, engine, read_ext, sheet_name, idx_lvl2):
        xfail_datetimes_with_pyxlsb(engine, request)
        mi_file = 'testmultiindex' + read_ext
        mi = MultiIndex.from_product([['foo', 'bar'], ['a', 'b']], names=['c1', 'c2'])
        unit = get_exp_unit(read_ext, engine)
        expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [2, 3.5, pd.Timestamp('2015-01-02'), False], [3, 4.5, pd.Timestamp('2015-01-03'), False], [4, 5.5, pd.Timestamp('2015-01-04'), True]], columns=mi, index=MultiIndex.from_arrays((['foo', 'foo', 'bar', 'bar'], idx_lvl2), names=['ilvl1', 'ilvl2']))
        expected[mi[2]] = expected[mi[2]].astype(f'M8[{unit}]')
        result = pd.read_excel(mi_file, sheet_name=sheet_name, index_col=[0, 1], header=[0, 1])
        tm.assert_frame_equal(result, expected)

    def test_read_excel_multiindex_header_only(self, read_ext):
        mi_file = 'testmultiindex' + read_ext
        result = pd.read_excel(mi_file, sheet_name='index_col_none', header=[0, 1])
        exp_columns = MultiIndex.from_product([('A', 'B'), ('key', 'val')])
        expected = DataFrame([[1, 2, 3, 4]] * 2, columns=exp_columns)
        tm.assert_frame_equal(result, expected)

    def test_excel_old_index_format(self, read_ext):
        filename = 'test_index_name_pre17' + read_ext
        data = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], ['R0C0', 'R0C1', 'R0C2', 'R0C3', 'R0C4'], ['R1C0', 'R1C1', 'R1C2', 'R1C3', 'R1C4'], ['R2C0', 'R2C1', 'R2C2', 'R2C3', 'R2C4'], ['R3C0', 'R3C1', 'R3C2', 'R3C3', 'R3C4'], ['R4C0', 'R4C1', 'R4C2', 'R4C3', 'R4C4']], dtype=object)
        columns = ['C_l0_g0', 'C_l0_g1', 'C_l0_g2', 'C_l0_g3', 'C_l0_g4']
        mi = MultiIndex(levels=[['R0', 'R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3', 'R_l0_g4'], ['R1', 'R_l1_g0', 'R_l1_g1', 'R_l1_g2', 'R_l1_g3', 'R_l1_g4']], codes=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], names=[None, None])
        si = Index(['R0', 'R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3', 'R_l0_g4'], name=None)
        expected = DataFrame(data, index=si, columns=columns)
        actual = pd.read_excel(filename, sheet_name='single_names', index_col=0)
        tm.assert_frame_equal(actual, expected)
        expected.index = mi
        actual = pd.read_excel(filename, sheet_name='multi_names', index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)
        data = np.array([['R0C0', 'R0C1', 'R0C2', 'R0C3', 'R0C4'], ['R1C0', 'R1C1', 'R1C2', 'R1C3', 'R1C4'], ['R2C0', 'R2C1', 'R2C2', 'R2C3', 'R2C4'], ['R3C0', 'R3C1', 'R3C2', 'R3C3', 'R3C4'], ['R4C0', 'R4C1', 'R4C2', 'R4C3', 'R4C4']])
        columns = ['C_l0_g0', 'C_l0_g1', 'C_l0_g2', 'C_l0_g3', 'C_l0_g4']
        mi = MultiIndex(levels=[['R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3', 'R_l0_g4'], ['R_l1_g0', 'R_l1_g1', 'R_l1_g2', 'R_l1_g3', 'R_l1_g4']], codes=[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], names=[None, None])
        si = Index(['R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3', 'R_l0_g4'], name=None)
        expected = DataFrame(data, index=si, columns=columns)
        actual = pd.read_excel(filename, sheet_name='single_no_names', index_col=0)
        tm.assert_frame_equal(actual, expected)
        expected.index = mi
        actual = pd.read_excel(filename, sheet_name='multi_no_names', index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_bool_header_arg(self, read_ext):
        msg = 'Passing a bool to header is invalid'
        for arg in [True, False]:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel('test1' + read_ext, header=arg)

    def test_read_excel_skiprows(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)
        unit = get_exp_unit(read_ext, engine)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=[0, 2])
        expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [2, 3.5, pd.Timestamp('2015-01-02'), False], [3, 4.5, pd.Timestamp('2015-01-03'), False], [4, 5.5, pd.Timestamp('2015-01-04'), True]], columns=['a', 'b', 'c', 'd'])
        expected['c'] = expected['c'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=np.array([0, 2]))
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=lambda x: x in [0, 2])
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=3, names=['a', 'b', 'c', 'd'])
        expected = DataFrame([[2, 3.5, pd.Timestamp('2015-01-02'), False], [3, 4.5, pd.Timestamp('2015-01-03'), False], [4, 5.5, pd.Timestamp('2015-01-04'), True]], columns=['a', 'b', 'c', 'd'])
        expected['c'] = expected['c'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_skiprows_callable_not_in(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)
        unit = get_exp_unit(read_ext, engine)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=lambda x: x not in [1, 3, 5])
        expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [3, 4.5, pd.Timestamp('2015-01-03'), False]], columns=['a', 'b', 'c', 'd'])
        expected['c'] = expected['c'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows(self, read_ext):
        num_rows_to_pull = 5
        actual = pd.read_excel('test1' + read_ext, nrows=num_rows_to_pull)
        expected = pd.read_excel('test1' + read_ext)
        expected = expected[:num_rows_to_pull]
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows_greater_than_nrows_in_file(self, read_ext):
        expected = pd.read_excel('test1' + read_ext)
        num_records_in_file = len(expected)
        num_rows_to_pull = num_records_in_file + 10
        actual = pd.read_excel('test1' + read_ext, nrows=num_rows_to_pull)
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows_non_integer_parameter(self, read_ext):
        msg = "'nrows' must be an integer >=0"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, nrows='5')

    @pytest.mark.parametrize('filename,sheet_name,header,index_col,skiprows', [('testmultiindex', 'mi_column', [0, 1], 0, None), ('testmultiindex', 'mi_index', None, [0, 1], None), ('testmultiindex', 'both', [0, 1], [0, 1], None), ('testmultiindex', 'mi_column_name', [0, 1], 0, None), ('testskiprows', 'skiprows_list', None, None, [0, 2]), ('testskiprows', 'skiprows_list', None, None, lambda x: x in (0, 2))])
    def test_read_excel_nrows_params(self, read_ext, filename, sheet_name, header, index_col, skiprows):
        """
        For various parameters, we should get the same result whether we
        limit the rows during load (nrows=3) or after (df.iloc[:3]).
        """
        expected = pd.read_excel(filename + read_ext, sheet_name=sheet_name, header=header, index_col=index_col, skiprows=skiprows).iloc[:3]
        actual = pd.read_excel(filename + read_ext, sheet_name=sheet_name, header=header, index_col=index_col, skiprows=skiprows, nrows=3)
        tm.assert_frame_equal(actual, expected)

    def test_deprecated_kwargs(self, read_ext):
        with pytest.raises(TypeError, match='but 3 positional arguments'):
            pd.read_excel('test1' + read_ext, 'Sheet1', 0)

    def test_no_header_with_list_index_col(self, read_ext):
        file_name = 'testmultiindex' + read_ext
        data = [('B', 'B'), ('key', 'val'), (3, 4), (3, 4)]
        idx = MultiIndex.from_tuples([('A', 'A'), ('key', 'val'), (1, 2), (1, 2)], names=(0, 1))
        expected = DataFrame(data, index=idx, columns=(2, 3))
        result = pd.read_excel(file_name, sheet_name='index_col_none', index_col=[0, 1], header=None)
        tm.assert_frame_equal(expected, result)

    def test_one_col_noskip_blank_line(self, read_ext):
        file_name = 'one_col_blank_line' + read_ext
        data = [0.5, np.nan, 1, 2]
        expected = DataFrame(data, columns=['numbers'])
        result = pd.read_excel(file_name)
        tm.assert_frame_equal(result, expected)

    def test_multiheader_two_blank_lines(self, read_ext):
        file_name = 'testmultiindex' + read_ext
        columns = MultiIndex.from_tuples([('a', 'A'), ('b', 'B')])
        data = [[np.nan, np.nan], [np.nan, np.nan], [1, 3], [2, 4]]
        expected = DataFrame(data, columns=columns)
        result = pd.read_excel(file_name, sheet_name='mi_column_empty_rows', header=[0, 1])
        tm.assert_frame_equal(result, expected)

    def test_trailing_blanks(self, read_ext):
        """
        Sheets can contain blank cells with no data. Some of our readers
        were including those cells, creating many empty rows and columns
        """
        file_name = 'trailing_blanks' + read_ext
        result = pd.read_excel(file_name)
        assert result.shape == (3, 3)

    def test_ignore_chartsheets_by_str(self, request, engine, read_ext):
        if read_ext == '.ods':
            pytest.skip('chartsheets do not exist in the ODF format')
        if engine == 'pyxlsb':
            request.applymarker(pytest.mark.xfail(reason="pyxlsb can't distinguish chartsheets from worksheets"))
        with pytest.raises(ValueError, match="Worksheet named 'Chart1' not found"):
            pd.read_excel('chartsheet' + read_ext, sheet_name='Chart1')

    def test_ignore_chartsheets_by_int(self, request, engine, read_ext):
        if read_ext == '.ods':
            pytest.skip('chartsheets do not exist in the ODF format')
        if engine == 'pyxlsb':
            request.applymarker(pytest.mark.xfail(reason="pyxlsb can't distinguish chartsheets from worksheets"))
        with pytest.raises(ValueError, match='Worksheet index 1 is invalid, 1 worksheets found'):
            pd.read_excel('chartsheet' + read_ext, sheet_name=1)

    def test_euro_decimal_format(self, read_ext):
        result = pd.read_excel('test_decimal' + read_ext, decimal=',', skiprows=1)
        expected = DataFrame([[1, 1521.1541, 187101.9543, 'ABC', 'poi', 4.738797819], [2, 121.12, 14897.76, 'DEF', 'uyt', 0.377320872], [3, 878.158, 108013.434, 'GHI', 'rez', 2.735694704]], columns=['Id', 'Number1', 'Number2', 'Text1', 'Text2', 'Number3'])
        tm.assert_frame_equal(result, expected)