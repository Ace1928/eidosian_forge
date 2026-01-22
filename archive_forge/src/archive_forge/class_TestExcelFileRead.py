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
class TestExcelFileRead:

    def test_deprecate_bytes_input(self, engine, read_ext):
        msg = "Passing bytes to 'read_excel' is deprecated and will be removed in a future version. To read from a byte string, wrap it in a `BytesIO` object."
        with tm.assert_produces_warning(FutureWarning, match=msg, raise_on_extra_warnings=False):
            with open('test1' + read_ext, 'rb') as f:
                pd.read_excel(f.read(), engine=engine)

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine, datapath, monkeypatch):
        """
        Change directory and set engine for ExcelFile objects.
        """
        func = partial(pd.ExcelFile, engine=engine)
        monkeypatch.chdir(datapath('io', 'data', 'excel'))
        monkeypatch.setattr(pd, 'ExcelFile', func)

    def test_engine_used(self, read_ext, engine):
        expected_defaults = {'xlsx': 'openpyxl', 'xlsm': 'openpyxl', 'xlsb': 'pyxlsb', 'xls': 'xlrd', 'ods': 'odf'}
        with pd.ExcelFile('test1' + read_ext) as excel:
            result = excel.engine
        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_excel_passes_na(self, read_ext):
        with pd.ExcelFile('test4' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=False, na_values=['apple'])
        expected = DataFrame([['NA'], [1], ['NA'], [np.nan], ['rabbit']], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile('test4' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=True, na_values=['apple'])
        expected = DataFrame([[np.nan], [1], [np.nan], [np.nan], ['rabbit']], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile('test5' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=False, na_values=['apple'])
        expected = DataFrame([['1.#QNAN'], [1], ['nan'], [np.nan], ['rabbit']], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile('test5' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=True, na_values=['apple'])
        expected = DataFrame([[np.nan], [1], [np.nan], [np.nan], ['rabbit']], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('na_filter', [None, True, False])
    def test_excel_passes_na_filter(self, read_ext, na_filter):
        kwargs = {}
        if na_filter is not None:
            kwargs['na_filter'] = na_filter
        with pd.ExcelFile('test5' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=True, na_values=['apple'], **kwargs)
        if na_filter is False:
            expected = [['1.#QNAN'], [1], ['nan'], ['apple'], ['rabbit']]
        else:
            expected = [[np.nan], [1], [np.nan], [np.nan], ['rabbit']]
        expected = DataFrame(expected, columns=['Test'])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_table_sheet_by_index(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)
        expected = df_ref
        adjust_expected(expected, read_ext, engine)
        with pd.ExcelFile('test1' + read_ext) as excel:
            df1 = pd.read_excel(excel, sheet_name=0, index_col=0)
            df2 = pd.read_excel(excel, sheet_name=1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        with pd.ExcelFile('test1' + read_ext) as excel:
            df1 = excel.parse(0, index_col=0)
            df2 = excel.parse(1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        with pd.ExcelFile('test1' + read_ext) as excel:
            df3 = pd.read_excel(excel, sheet_name=0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])
        with pd.ExcelFile('test1' + read_ext) as excel:
            df3 = excel.parse(0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def test_sheet_name(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)
        expected = df_ref
        adjust_expected(expected, read_ext, engine)
        filename = 'test1'
        sheet_name = 'Sheet1'
        with pd.ExcelFile(filename + read_ext) as excel:
            df1_parse = excel.parse(sheet_name=sheet_name, index_col=0)
        with pd.ExcelFile(filename + read_ext) as excel:
            df2_parse = excel.parse(index_col=0, sheet_name=sheet_name)
        tm.assert_frame_equal(df1_parse, expected)
        tm.assert_frame_equal(df2_parse, expected)

    @pytest.mark.parametrize('sheet_name', [3, [0, 3], [3, 0], 'Sheet4', ['Sheet1', 'Sheet4'], ['Sheet4', 'Sheet1']])
    def test_bad_sheetname_raises(self, read_ext, sheet_name):
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            with pd.ExcelFile('blank' + read_ext) as excel:
                excel.parse(sheet_name=sheet_name)

    def test_excel_read_buffer(self, engine, read_ext):
        pth = 'test1' + read_ext
        expected = pd.read_excel(pth, sheet_name='Sheet1', index_col=0, engine=engine)
        with open(pth, 'rb') as f:
            with pd.ExcelFile(f) as xls:
                actual = pd.read_excel(xls, sheet_name='Sheet1', index_col=0)
        tm.assert_frame_equal(expected, actual)

    def test_reader_closes_file(self, engine, read_ext):
        with open('test1' + read_ext, 'rb') as f:
            with pd.ExcelFile(f) as xlsx:
                pd.read_excel(xlsx, sheet_name='Sheet1', index_col=0, engine=engine)
        assert f.closed

    def test_conflicting_excel_engines(self, read_ext):
        msg = 'Engine should not be specified when passing an ExcelFile'
        with pd.ExcelFile('test1' + read_ext) as xl:
            with pytest.raises(ValueError, match=msg):
                pd.read_excel(xl, engine='foo')

    def test_excel_read_binary(self, engine, read_ext):
        expected = pd.read_excel('test1' + read_ext, engine=engine)
        with open('test1' + read_ext, 'rb') as f:
            data = f.read()
        actual = pd.read_excel(BytesIO(data), engine=engine)
        tm.assert_frame_equal(expected, actual)

    def test_excel_read_binary_via_read_excel(self, read_ext, engine):
        with open('test1' + read_ext, 'rb') as f:
            result = pd.read_excel(f, engine=engine)
        expected = pd.read_excel('test1' + read_ext, engine=engine)
        tm.assert_frame_equal(result, expected)

    def test_read_excel_header_index_out_of_range(self, engine):
        with open('df_header_oob.xlsx', 'rb') as f:
            with pytest.raises(ValueError, match='exceeds maximum'):
                pd.read_excel(f, header=[0, 1])

    @pytest.mark.parametrize('filename', ['df_empty.xlsx', 'df_equals.xlsx'])
    def test_header_with_index_col(self, filename):
        idx = Index(['Z'], name='I2')
        cols = MultiIndex.from_tuples([('A', 'B'), ('A', 'B.1')], names=['I11', 'I12'])
        expected = DataFrame([[1, 3]], index=idx, columns=cols, dtype='int64')
        result = pd.read_excel(filename, sheet_name='Sheet1', index_col=0, header=[0, 1])
        tm.assert_frame_equal(expected, result)

    def test_read_datetime_multiindex(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)
        f = 'test_datetime_mi' + read_ext
        with pd.ExcelFile(f) as excel:
            actual = pd.read_excel(excel, header=[0, 1], index_col=0, engine=engine)
        unit = get_exp_unit(read_ext, engine)
        dti = pd.DatetimeIndex(['2020-02-29', '2020-03-01'], dtype=f'M8[{unit}]')
        expected_column_index = MultiIndex.from_arrays([dti[:1], dti[1:]], names=[dti[0].to_pydatetime(), dti[1].to_pydatetime()])
        expected = DataFrame([], index=[], columns=expected_column_index)
        tm.assert_frame_equal(expected, actual)

    def test_engine_invalid_option(self, read_ext):
        with pytest.raises(ValueError, match='Value must be one of *'):
            with pd.option_context(f'io.excel{read_ext}.reader', 'abc'):
                pass

    def test_ignore_chartsheets(self, request, engine, read_ext):
        if read_ext == '.ods':
            pytest.skip('chartsheets do not exist in the ODF format')
        if engine == 'pyxlsb':
            request.applymarker(pytest.mark.xfail(reason="pyxlsb can't distinguish chartsheets from worksheets"))
        with pd.ExcelFile('chartsheet' + read_ext) as excel:
            assert excel.sheet_names == ['Sheet1']

    def test_corrupt_files_closed(self, engine, read_ext):
        errors = (BadZipFile,)
        if engine is None:
            pytest.skip(f'Invalid test for engine={engine}')
        elif engine == 'xlrd':
            import xlrd
            errors = (BadZipFile, xlrd.biffh.XLRDError)
        elif engine == 'calamine':
            from python_calamine import CalamineError
            errors = (CalamineError,)
        with tm.ensure_clean(f'corrupt{read_ext}') as file:
            Path(file).write_text('corrupt', encoding='utf-8')
            with tm.assert_produces_warning(False):
                try:
                    pd.ExcelFile(file, engine=engine)
                except errors:
                    pass