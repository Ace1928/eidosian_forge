from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
@pytest.mark.parametrize('engine,ext', [pytest.param('openpyxl', '.xlsx', marks=[td.skip_if_no('openpyxl'), td.skip_if_no('xlrd')]), pytest.param('openpyxl', '.xlsm', marks=[td.skip_if_no('openpyxl'), td.skip_if_no('xlrd')]), pytest.param('xlsxwriter', '.xlsx', marks=[td.skip_if_no('xlsxwriter'), td.skip_if_no('xlrd')]), pytest.param('odf', '.ods', marks=td.skip_if_no('odf'))])
@pytest.mark.usefixtures('set_engine')
class TestExcelWriter:

    def test_excel_sheet_size(self, path):
        breaking_row_count = 2 ** 20 + 1
        breaking_col_count = 2 ** 14 + 1
        row_arr = np.zeros(shape=(breaking_row_count, 1))
        col_arr = np.zeros(shape=(1, breaking_col_count))
        row_df = DataFrame(row_arr)
        col_df = DataFrame(col_arr)
        msg = 'sheet is too large'
        with pytest.raises(ValueError, match=msg):
            row_df.to_excel(path)
        with pytest.raises(ValueError, match=msg):
            col_df.to_excel(path)

    def test_excel_sheet_by_name_raise(self, path):
        gt = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        gt.to_excel(path)
        with ExcelFile(path) as xl:
            df = pd.read_excel(xl, sheet_name=0, index_col=0)
        tm.assert_frame_equal(gt, df)
        msg = "Worksheet named '0' not found"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(xl, '0')

    def test_excel_writer_context_manager(self, frame, path):
        with ExcelWriter(path) as writer:
            frame.to_excel(writer, sheet_name='Data1')
            frame2 = frame.copy()
            frame2.columns = frame.columns[::-1]
            frame2.to_excel(writer, sheet_name='Data2')
        with ExcelFile(path) as reader:
            found_df = pd.read_excel(reader, sheet_name='Data1', index_col=0)
            found_df2 = pd.read_excel(reader, sheet_name='Data2', index_col=0)
            tm.assert_frame_equal(found_df, frame)
            tm.assert_frame_equal(found_df2, frame2)

    def test_roundtrip(self, frame, path):
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.to_excel(path, sheet_name='test1')
        frame.to_excel(path, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(path, sheet_name='test1', header=False)
        frame.to_excel(path, sheet_name='test1', index=False)
        frame.to_excel(path, sheet_name='test1')
        recons = pd.read_excel(path, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(path, sheet_name='test1', index=False)
        recons = pd.read_excel(path, sheet_name='test1', index_col=None)
        recons.index = frame.index
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(path, sheet_name='test1', na_rep='NA')
        recons = pd.read_excel(path, sheet_name='test1', index_col=0, na_values=['NA'])
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(path, sheet_name='test1', na_rep='88')
        recons = pd.read_excel(path, sheet_name='test1', index_col=0, na_values=['88'])
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(path, sheet_name='test1', na_rep='88')
        recons = pd.read_excel(path, sheet_name='test1', index_col=0, na_values=[88, 88.0])
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(path, sheet_name='Sheet1')
        recons = pd.read_excel(path, index_col=0)
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(path, sheet_name='0')
        recons = pd.read_excel(path, index_col=0)
        tm.assert_frame_equal(frame, recons)
        s = frame['A']
        s.to_excel(path)
        recons = pd.read_excel(path, index_col=0)
        tm.assert_frame_equal(s.to_frame(), recons)

    def test_mixed(self, frame, path):
        mixed_frame = frame.copy()
        mixed_frame['foo'] = 'bar'
        mixed_frame.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(mixed_frame, recons)

    def test_ts_frame(self, path):
        unit = get_exp_unit(path)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD')), index=date_range('2000-01-01', periods=5, freq='B'))
        index = pd.DatetimeIndex(np.asarray(df.index), freq=None)
        df.index = index
        expected = df[:]
        expected.index = expected.index.as_unit(unit)
        df.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(expected, recons)

    def test_basics_with_nan(self, frame, path):
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.to_excel(path, sheet_name='test1')
        frame.to_excel(path, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(path, sheet_name='test1', header=False)
        frame.to_excel(path, sheet_name='test1', index=False)

    @pytest.mark.parametrize('np_type', [np.int8, np.int16, np.int32, np.int64])
    def test_int_types(self, np_type, path):
        df = DataFrame(np.random.default_rng(2).integers(-10, 10, size=(10, 2)), dtype=np_type)
        df.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
        int_frame = df.astype(np.int64)
        tm.assert_frame_equal(int_frame, recons)
        recons2 = pd.read_excel(path, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(int_frame, recons2)

    @pytest.mark.parametrize('np_type', [np.float16, np.float32, np.float64])
    def test_float_types(self, np_type, path):
        df = DataFrame(np.random.default_rng(2).random(10), dtype=np_type)
        df.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0).astype(np_type)
        tm.assert_frame_equal(df, recons)

    def test_bool_types(self, path):
        df = DataFrame([1, 0, True, False], dtype=np.bool_)
        df.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0).astype(np.bool_)
        tm.assert_frame_equal(df, recons)

    def test_inf_roundtrip(self, path):
        df = DataFrame([(1, np.inf), (2, 3), (5, -np.inf)])
        df.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(df, recons)

    def test_sheets(self, frame, path):
        unit = get_exp_unit(path)
        tsframe = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD')), index=date_range('2000-01-01', periods=5, freq='B'))
        index = pd.DatetimeIndex(np.asarray(tsframe.index), freq=None)
        tsframe.index = index
        expected = tsframe[:]
        expected.index = expected.index.as_unit(unit)
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.to_excel(path, sheet_name='test1')
        frame.to_excel(path, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(path, sheet_name='test1', header=False)
        frame.to_excel(path, sheet_name='test1', index=False)
        with ExcelWriter(path) as writer:
            frame.to_excel(writer, sheet_name='test1')
            tsframe.to_excel(writer, sheet_name='test2')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
            tm.assert_frame_equal(frame, recons)
            recons = pd.read_excel(reader, sheet_name='test2', index_col=0)
        tm.assert_frame_equal(expected, recons)
        assert 2 == len(reader.sheet_names)
        assert 'test1' == reader.sheet_names[0]
        assert 'test2' == reader.sheet_names[1]

    def test_colaliases(self, frame, path):
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.to_excel(path, sheet_name='test1')
        frame.to_excel(path, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(path, sheet_name='test1', header=False)
        frame.to_excel(path, sheet_name='test1', index=False)
        col_aliases = Index(['AA', 'X', 'Y', 'Z'])
        frame.to_excel(path, sheet_name='test1', header=col_aliases)
        with ExcelFile(path) as reader:
            rs = pd.read_excel(reader, sheet_name='test1', index_col=0)
        xp = frame.copy()
        xp.columns = col_aliases
        tm.assert_frame_equal(xp, rs)

    def test_roundtrip_indexlabels(self, merge_cells, frame, path):
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.to_excel(path, sheet_name='test1')
        frame.to_excel(path, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(path, sheet_name='test1', header=False)
        frame.to_excel(path, sheet_name='test1', index=False)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2))) >= 0
        df.to_excel(path, sheet_name='test1', index_label=['test'], merge_cells=merge_cells)
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0).astype(np.int64)
        df.index.names = ['test']
        assert df.index.names == recons.index.names
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2))) >= 0
        df.to_excel(path, sheet_name='test1', index_label=['test', 'dummy', 'dummy2'], merge_cells=merge_cells)
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0).astype(np.int64)
        df.index.names = ['test']
        assert df.index.names == recons.index.names
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2))) >= 0
        df.to_excel(path, sheet_name='test1', index_label='test', merge_cells=merge_cells)
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0).astype(np.int64)
        df.index.names = ['test']
        tm.assert_frame_equal(df, recons.astype(bool))
        frame.to_excel(path, sheet_name='test1', columns=['A', 'B', 'C', 'D'], index=False, merge_cells=merge_cells)
        df = frame.copy()
        df = df.set_index(['A', 'B'])
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=[0, 1])
        tm.assert_frame_equal(df, recons)

    def test_excel_roundtrip_indexname(self, merge_cells, path):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        df.index.name = 'foo'
        df.to_excel(path, merge_cells=merge_cells)
        with ExcelFile(path) as xf:
            result = pd.read_excel(xf, sheet_name=xf.sheet_names[0], index_col=0)
        tm.assert_frame_equal(result, df)
        assert result.index.name == 'foo'

    def test_excel_roundtrip_datetime(self, merge_cells, path):
        unit = get_exp_unit(path)
        tsframe = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD')), index=date_range('2000-01-01', periods=5, freq='B'))
        index = pd.DatetimeIndex(np.asarray(tsframe.index), freq=None)
        tsframe.index = index
        tsf = tsframe.copy()
        tsf.index = [x.date() for x in tsframe.index]
        tsf.to_excel(path, sheet_name='test1', merge_cells=merge_cells)
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
        expected = tsframe[:]
        expected.index = expected.index.as_unit(unit)
        tm.assert_frame_equal(expected, recons)

    def test_excel_date_datetime_format(self, ext, path):
        unit = get_exp_unit(path)
        df = DataFrame([[date(2014, 1, 31), date(1999, 9, 24)], [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)]], index=['DATE', 'DATETIME'], columns=['X', 'Y'])
        df_expected = DataFrame([[datetime(2014, 1, 31), datetime(1999, 9, 24)], [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)]], index=['DATE', 'DATETIME'], columns=['X', 'Y'])
        df_expected = df_expected.astype(f'M8[{unit}]')
        with tm.ensure_clean(ext) as filename2:
            with ExcelWriter(path) as writer1:
                df.to_excel(writer1, sheet_name='test1')
            with ExcelWriter(filename2, date_format='DD.MM.YYYY', datetime_format='DD.MM.YYYY HH-MM-SS') as writer2:
                df.to_excel(writer2, sheet_name='test1')
            with ExcelFile(path) as reader1:
                rs1 = pd.read_excel(reader1, sheet_name='test1', index_col=0)
            with ExcelFile(filename2) as reader2:
                rs2 = pd.read_excel(reader2, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(rs1, rs2)
        tm.assert_frame_equal(rs2, df_expected)

    def test_to_excel_interval_no_labels(self, path, using_infer_string):
        df = DataFrame(np.random.default_rng(2).integers(-10, 10, size=(20, 1)), dtype=np.int64)
        expected = df.copy()
        df['new'] = pd.cut(df[0], 10)
        expected['new'] = pd.cut(expected[0], 10).astype(str if not using_infer_string else 'string[pyarrow_numpy]')
        df.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(expected, recons)

    def test_to_excel_interval_labels(self, path):
        df = DataFrame(np.random.default_rng(2).integers(-10, 10, size=(20, 1)), dtype=np.int64)
        expected = df.copy()
        intervals = pd.cut(df[0], 10, labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        df['new'] = intervals
        expected['new'] = pd.Series(list(intervals))
        df.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(expected, recons)

    def test_to_excel_timedelta(self, path):
        df = DataFrame(np.random.default_rng(2).integers(-10, 10, size=(20, 1)), columns=['A'], dtype=np.int64)
        expected = df.copy()
        df['new'] = df['A'].apply(lambda x: timedelta(seconds=x))
        expected['new'] = expected['A'].apply(lambda x: timedelta(seconds=x).total_seconds() / 86400)
        df.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(expected, recons)

    def test_to_excel_periodindex(self, path):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD')), index=date_range('2000-01-01', periods=5, freq='B'))
        xp = df.resample('ME').mean().to_period('M')
        xp.to_excel(path, sheet_name='sht1')
        with ExcelFile(path) as reader:
            rs = pd.read_excel(reader, sheet_name='sht1', index_col=0)
        tm.assert_frame_equal(xp, rs.to_period('M'))

    def test_to_excel_multiindex(self, merge_cells, frame, path):
        arrays = np.arange(len(frame.index) * 2, dtype=np.int64).reshape(2, -1)
        new_index = MultiIndex.from_arrays(arrays, names=['first', 'second'])
        frame.index = new_index
        frame.to_excel(path, sheet_name='test1', header=False)
        frame.to_excel(path, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(path, sheet_name='test1', merge_cells=merge_cells)
        with ExcelFile(path) as reader:
            df = pd.read_excel(reader, sheet_name='test1', index_col=[0, 1])
        tm.assert_frame_equal(frame, df)

    def test_to_excel_multiindex_nan_label(self, merge_cells, path):
        df = DataFrame({'A': [None, 2, 3], 'B': [10, 20, 30], 'C': np.random.default_rng(2).random(3)})
        df = df.set_index(['A', 'B'])
        df.to_excel(path, merge_cells=merge_cells)
        df1 = pd.read_excel(path, index_col=[0, 1])
        tm.assert_frame_equal(df, df1)

    def test_to_excel_multiindex_cols(self, merge_cells, frame, path):
        arrays = np.arange(len(frame.index) * 2, dtype=np.int64).reshape(2, -1)
        new_index = MultiIndex.from_arrays(arrays, names=['first', 'second'])
        frame.index = new_index
        new_cols_index = MultiIndex.from_tuples([(40, 1), (40, 2), (50, 1), (50, 2)])
        frame.columns = new_cols_index
        header = [0, 1]
        if not merge_cells:
            header = 0
        frame.to_excel(path, sheet_name='test1', merge_cells=merge_cells)
        with ExcelFile(path) as reader:
            df = pd.read_excel(reader, sheet_name='test1', header=header, index_col=[0, 1])
        if not merge_cells:
            fm = frame.columns._format_multi(sparsify=False, include_names=False)
            frame.columns = ['.'.join(map(str, q)) for q in zip(*fm)]
        tm.assert_frame_equal(frame, df)

    def test_to_excel_multiindex_dates(self, merge_cells, path):
        unit = get_exp_unit(path)
        tsframe = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD')), index=date_range('2000-01-01', periods=5, freq='B'))
        tsframe.index = MultiIndex.from_arrays([tsframe.index.as_unit(unit), np.arange(len(tsframe.index), dtype=np.int64)], names=['time', 'foo'])
        tsframe.to_excel(path, sheet_name='test1', merge_cells=merge_cells)
        with ExcelFile(path) as reader:
            recons = pd.read_excel(reader, sheet_name='test1', index_col=[0, 1])
        tm.assert_frame_equal(tsframe, recons)
        assert recons.index.names == ('time', 'foo')

    def test_to_excel_multiindex_no_write_index(self, path):
        frame1 = DataFrame({'a': [10, 20], 'b': [30, 40], 'c': [50, 60]})
        frame2 = frame1.copy()
        multi_index = MultiIndex.from_tuples([(70, 80), (90, 100)])
        frame2.index = multi_index
        frame2.to_excel(path, sheet_name='test1', index=False)
        with ExcelFile(path) as reader:
            frame3 = pd.read_excel(reader, sheet_name='test1')
        tm.assert_frame_equal(frame1, frame3)

    def test_to_excel_empty_multiindex(self, path):
        expected = DataFrame([], columns=[0, 1, 2])
        df = DataFrame([], index=MultiIndex.from_tuples([], names=[0, 1]), columns=[2])
        df.to_excel(path, sheet_name='test1')
        with ExcelFile(path) as reader:
            result = pd.read_excel(reader, sheet_name='test1')
        tm.assert_frame_equal(result, expected, check_index_type=False, check_dtype=False)

    def test_to_excel_float_format(self, path):
        df = DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
        df.to_excel(path, sheet_name='test1', float_format='%.2f')
        with ExcelFile(path) as reader:
            result = pd.read_excel(reader, sheet_name='test1', index_col=0)
        expected = DataFrame([[0.12, 0.23, 0.57], [12.32, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
        tm.assert_frame_equal(result, expected)

    def test_to_excel_output_encoding(self, ext):
        df = DataFrame([['ƒ', 'Ɠ', 'Ɣ'], ['ƕ', 'Ɩ', 'Ɨ']], index=['Aƒ', 'B'], columns=['XƓ', 'Y', 'Z'])
        with tm.ensure_clean('__tmp_to_excel_float_format__.' + ext) as filename:
            df.to_excel(filename, sheet_name='TestSheet')
            result = pd.read_excel(filename, sheet_name='TestSheet', index_col=0)
            tm.assert_frame_equal(result, df)

    def test_to_excel_unicode_filename(self, ext):
        with tm.ensure_clean('ƒu.' + ext) as filename:
            try:
                with open(filename, 'wb'):
                    pass
            except UnicodeEncodeError:
                pytest.skip('No unicode file names on this system')
            df = DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
            df.to_excel(filename, sheet_name='test1', float_format='%.2f')
            with ExcelFile(filename) as reader:
                result = pd.read_excel(reader, sheet_name='test1', index_col=0)
        expected = DataFrame([[0.12, 0.23, 0.57], [12.32, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('use_headers', [True, False])
    @pytest.mark.parametrize('r_idx_nlevels', [1, 2, 3])
    @pytest.mark.parametrize('c_idx_nlevels', [1, 2, 3])
    def test_excel_010_hemstring(self, merge_cells, c_idx_nlevels, r_idx_nlevels, use_headers, path):

        def roundtrip(data, header=True, parser_hdr=0, index=True):
            data.to_excel(path, header=header, merge_cells=merge_cells, index=index)
            with ExcelFile(path) as xf:
                return pd.read_excel(xf, sheet_name=xf.sheet_names[0], header=parser_hdr)
        parser_header = 0 if use_headers else None
        res = roundtrip(DataFrame([0]), use_headers, parser_header)
        assert res.shape == (1, 2)
        assert res.iloc[0, 0] is not np.nan
        nrows = 5
        ncols = 3
        if c_idx_nlevels == 1:
            columns = Index([f'a-{i}' for i in range(ncols)], dtype=object)
        else:
            columns = MultiIndex.from_arrays([range(ncols) for _ in range(c_idx_nlevels)], names=[f'i-{i}' for i in range(c_idx_nlevels)])
        if r_idx_nlevels == 1:
            index = Index([f'b-{i}' for i in range(nrows)], dtype=object)
        else:
            index = MultiIndex.from_arrays([range(nrows) for _ in range(r_idx_nlevels)], names=[f'j-{i}' for i in range(r_idx_nlevels)])
        df = DataFrame(np.ones((nrows, ncols)), columns=columns, index=index)
        if c_idx_nlevels > 1:
            msg = "Writing to Excel with MultiIndex columns and no index \\('index'=False\\) is not yet implemented."
            with pytest.raises(NotImplementedError, match=msg):
                roundtrip(df, use_headers, index=False)
        else:
            res = roundtrip(df, use_headers)
            if use_headers:
                assert res.shape == (nrows, ncols + r_idx_nlevels)
            else:
                assert res.shape == (nrows - 1, ncols + r_idx_nlevels)
            for r in range(len(res.index)):
                for c in range(len(res.columns)):
                    assert res.iloc[r, c] is not np.nan

    def test_duplicated_columns(self, path):
        df = DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=['A', 'B', 'B'])
        df.to_excel(path, sheet_name='test1')
        expected = DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=['A', 'B', 'B.1'])
        result = pd.read_excel(path, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(result, expected)
        df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=['A', 'B', 'A', 'B'])
        df.to_excel(path, sheet_name='test1')
        result = pd.read_excel(path, sheet_name='test1', index_col=0)
        expected = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=['A', 'B', 'A.1', 'B.1'])
        tm.assert_frame_equal(result, expected)
        df.to_excel(path, sheet_name='test1', index=False, header=False)
        result = pd.read_excel(path, sheet_name='test1', header=None)
        expected = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
        tm.assert_frame_equal(result, expected)

    def test_swapped_columns(self, path):
        write_frame = DataFrame({'A': [1, 1, 1], 'B': [2, 2, 2]})
        write_frame.to_excel(path, sheet_name='test1', columns=['B', 'A'])
        read_frame = pd.read_excel(path, sheet_name='test1', header=0)
        tm.assert_series_equal(write_frame['A'], read_frame['A'])
        tm.assert_series_equal(write_frame['B'], read_frame['B'])

    def test_invalid_columns(self, path):
        write_frame = DataFrame({'A': [1, 1, 1], 'B': [2, 2, 2]})
        with pytest.raises(KeyError, match='Not all names specified'):
            write_frame.to_excel(path, sheet_name='test1', columns=['B', 'C'])
        with pytest.raises(KeyError, match="'passes columns are not ALL present dataframe'"):
            write_frame.to_excel(path, sheet_name='test1', columns=['C', 'D'])

    @pytest.mark.parametrize('to_excel_index,read_excel_index_col', [(True, 0), (False, None)])
    def test_write_subset_columns(self, path, to_excel_index, read_excel_index_col):
        write_frame = DataFrame({'A': [1, 1, 1], 'B': [2, 2, 2], 'C': [3, 3, 3]})
        write_frame.to_excel(path, sheet_name='col_subset_bug', columns=['A', 'B'], index=to_excel_index)
        expected = write_frame[['A', 'B']]
        read_frame = pd.read_excel(path, sheet_name='col_subset_bug', index_col=read_excel_index_col)
        tm.assert_frame_equal(expected, read_frame)

    def test_comment_arg(self, path):
        df = DataFrame({'A': ['one', '#one', 'one'], 'B': ['two', 'two', '#two']})
        df.to_excel(path, sheet_name='test_c')
        result1 = pd.read_excel(path, sheet_name='test_c', index_col=0)
        result1.iloc[1, 0] = None
        result1.iloc[1, 1] = None
        result1.iloc[2, 1] = None
        result2 = pd.read_excel(path, sheet_name='test_c', comment='#', index_col=0)
        tm.assert_frame_equal(result1, result2)

    def test_comment_default(self, path):
        df = DataFrame({'A': ['one', '#one', 'one'], 'B': ['two', 'two', '#two']})
        df.to_excel(path, sheet_name='test_c')
        result1 = pd.read_excel(path, sheet_name='test_c')
        result2 = pd.read_excel(path, sheet_name='test_c', comment=None)
        tm.assert_frame_equal(result1, result2)

    def test_comment_used(self, path):
        df = DataFrame({'A': ['one', '#one', 'one'], 'B': ['two', 'two', '#two']})
        df.to_excel(path, sheet_name='test_c')
        expected = DataFrame({'A': ['one', None, 'one'], 'B': ['two', None, None]})
        result = pd.read_excel(path, sheet_name='test_c', comment='#', index_col=0)
        tm.assert_frame_equal(result, expected)

    def test_comment_empty_line(self, path):
        df = DataFrame({'a': ['1', '#2'], 'b': ['2', '3']})
        df.to_excel(path, index=False)
        expected = DataFrame({'a': [1], 'b': [2]})
        result = pd.read_excel(path, comment='#')
        tm.assert_frame_equal(result, expected)

    def test_datetimes(self, path):
        unit = get_exp_unit(path)
        datetimes = [datetime(2013, 1, 13, 1, 2, 3), datetime(2013, 1, 13, 2, 45, 56), datetime(2013, 1, 13, 4, 29, 49), datetime(2013, 1, 13, 6, 13, 42), datetime(2013, 1, 13, 7, 57, 35), datetime(2013, 1, 13, 9, 41, 28), datetime(2013, 1, 13, 11, 25, 21), datetime(2013, 1, 13, 13, 9, 14), datetime(2013, 1, 13, 14, 53, 7), datetime(2013, 1, 13, 16, 37, 0), datetime(2013, 1, 13, 18, 20, 52)]
        write_frame = DataFrame({'A': datetimes})
        write_frame.to_excel(path, sheet_name='Sheet1')
        read_frame = pd.read_excel(path, sheet_name='Sheet1', header=0)
        expected = write_frame.astype(f'M8[{unit}]')
        tm.assert_series_equal(expected['A'], read_frame['A'])

    def test_bytes_io(self, engine):
        with BytesIO() as bio:
            df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
            with ExcelWriter(bio, engine=engine) as writer:
                df.to_excel(writer)
            bio.seek(0)
            reread_df = pd.read_excel(bio, index_col=0)
            tm.assert_frame_equal(df, reread_df)

    def test_engine_kwargs(self, engine, path):
        df = DataFrame([{'A': 1, 'B': 2}, {'A': 3, 'B': 4}])
        msgs = {'odf': "OpenDocumentSpreadsheet() got an unexpected keyword argument 'foo'", 'openpyxl': "__init__() got an unexpected keyword argument 'foo'", 'xlsxwriter': "__init__() got an unexpected keyword argument 'foo'"}
        if PY310:
            msgs['openpyxl'] = "Workbook.__init__() got an unexpected keyword argument 'foo'"
            msgs['xlsxwriter'] = "Workbook.__init__() got an unexpected keyword argument 'foo'"
        if engine == 'openpyxl' and (not os.path.exists(path)):
            msgs['openpyxl'] = "load_workbook() got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=re.escape(msgs[engine])):
            df.to_excel(path, engine=engine, engine_kwargs={'foo': 'bar'})

    def test_write_lists_dict(self, path):
        df = DataFrame({'mixed': ['a', ['b', 'c'], {'d': 'e', 'f': 2}], 'numeric': [1, 2, 3.0], 'str': ['apple', 'banana', 'cherry']})
        df.to_excel(path, sheet_name='Sheet1')
        read = pd.read_excel(path, sheet_name='Sheet1', header=0, index_col=0)
        expected = df.copy()
        expected.mixed = expected.mixed.apply(str)
        expected.numeric = expected.numeric.astype('int64')
        tm.assert_frame_equal(read, expected)

    def test_render_as_column_name(self, path):
        df = DataFrame({'render': [1, 2], 'data': [3, 4]})
        df.to_excel(path, sheet_name='Sheet1')
        read = pd.read_excel(path, 'Sheet1', index_col=0)
        expected = df
        tm.assert_frame_equal(read, expected)

    def test_true_and_false_value_options(self, path):
        df = DataFrame([['foo', 'bar']], columns=['col1', 'col2'], dtype=object)
        with option_context('future.no_silent_downcasting', True):
            expected = df.replace({'foo': True, 'bar': False}).astype('bool')
        df.to_excel(path)
        read_frame = pd.read_excel(path, true_values=['foo'], false_values=['bar'], index_col=0)
        tm.assert_frame_equal(read_frame, expected)

    def test_freeze_panes(self, path):
        expected = DataFrame([[1, 2], [3, 4]], columns=['col1', 'col2'])
        expected.to_excel(path, sheet_name='Sheet1', freeze_panes=(1, 1))
        result = pd.read_excel(path, index_col=0)
        tm.assert_frame_equal(result, expected)

    def test_path_path_lib(self, engine, ext):
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD')), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        writer = partial(df.to_excel, engine=engine)
        reader = partial(pd.read_excel, index_col=0)
        result = tm.round_trip_pathlib(writer, reader, path=f'foo{ext}')
        tm.assert_frame_equal(result, df)

    def test_path_local_path(self, engine, ext):
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD')), index=Index([f'i-{i}' for i in range(30)]))
        writer = partial(df.to_excel, engine=engine)
        reader = partial(pd.read_excel, index_col=0)
        result = tm.round_trip_localpath(writer, reader, path=f'foo{ext}')
        tm.assert_frame_equal(result, df)

    def test_merged_cell_custom_objects(self, path):
        mi = MultiIndex.from_tuples([(pd.Period('2018'), pd.Period('2018Q1')), (pd.Period('2018'), pd.Period('2018Q2'))])
        expected = DataFrame(np.ones((2, 2), dtype='int64'), columns=mi)
        expected.to_excel(path)
        result = pd.read_excel(path, header=[0, 1], index_col=0)
        expected.columns = expected.columns.set_levels([[str(i) for i in mi.levels[0]], [str(i) for i in mi.levels[1]]], level=[0, 1])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', [None, object])
    def test_raise_when_saving_timezones(self, dtype, tz_aware_fixture, path):
        tz = tz_aware_fixture
        data = pd.Timestamp('2019', tz=tz)
        df = DataFrame([data], dtype=dtype)
        with pytest.raises(ValueError, match='Excel does not support'):
            df.to_excel(path)
        data = data.to_pydatetime()
        df = DataFrame([data], dtype=dtype)
        with pytest.raises(ValueError, match='Excel does not support'):
            df.to_excel(path)

    def test_excel_duplicate_columns_with_names(self, path):
        df = DataFrame({'A': [0, 1], 'B': [10, 11]})
        df.to_excel(path, columns=['A', 'B', 'A'], index=False)
        result = pd.read_excel(path)
        expected = DataFrame([[0, 10, 0], [1, 11, 1]], columns=['A', 'B', 'A.1'])
        tm.assert_frame_equal(result, expected)

    def test_if_sheet_exists_raises(self, ext):
        msg = "if_sheet_exists is only valid in append mode (mode='a')"
        with tm.ensure_clean(ext) as f:
            with pytest.raises(ValueError, match=re.escape(msg)):
                ExcelWriter(f, if_sheet_exists='replace')

    def test_excel_writer_empty_frame(self, engine, ext):
        with tm.ensure_clean(ext) as path:
            with ExcelWriter(path, engine=engine) as writer:
                DataFrame().to_excel(writer)
            result = pd.read_excel(path)
            expected = DataFrame()
            tm.assert_frame_equal(result, expected)

    def test_to_excel_empty_frame(self, engine, ext):
        with tm.ensure_clean(ext) as path:
            DataFrame().to_excel(path, engine=engine)
            result = pd.read_excel(path)
            expected = DataFrame()
            tm.assert_frame_equal(result, expected)