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
class TestConvertMetadata:
    """
    Conversion tests for Pandas metadata & indices.
    """

    def test_non_string_columns(self):
        df = pd.DataFrame({0: [1, 2, 3]})
        table = pa.Table.from_pandas(df)
        assert table.field(0).name == '0'

    def test_non_string_columns_with_index(self):
        df = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]})
        df = df.set_index(0)
        with pytest.warns(UserWarning):
            table = pa.Table.from_pandas(df)
            assert table.field(0).name == '1'
        expected = df.copy()
        expected.index.name = str(expected.index.name)
        with pytest.warns(UserWarning):
            _check_pandas_roundtrip(df, expected=expected, preserve_index=True)

    def test_from_pandas_with_columns(self):
        df = pd.DataFrame({0: [1, 2, 3], 1: [1, 3, 3], 2: [2, 4, 5]}, columns=[1, 0])
        table = pa.Table.from_pandas(df, columns=[0, 1])
        expected = pa.Table.from_pandas(df[[0, 1]])
        assert expected.equals(table)
        record_batch_table = pa.RecordBatch.from_pandas(df, columns=[0, 1])
        record_batch_expected = pa.RecordBatch.from_pandas(df[[0, 1]])
        assert record_batch_expected.equals(record_batch_table)

    def test_column_index_names_are_preserved(self):
        df = pd.DataFrame({'data': [1, 2, 3]})
        df.columns.names = ['a']
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_column_index_names_with_tz(self):
        df = pd.DataFrame(np.random.randn(5, 3), columns=pd.date_range('2021-01-01', periods=3, freq='50D', tz='CET'))
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_range_index_shortcut(self):
        index_name = 'foo'
        df = pd.DataFrame({'a': [1, 2, 3, 4]}, index=pd.RangeIndex(0, 8, step=2, name=index_name))
        df2 = pd.DataFrame({'a': [4, 5, 6, 7]}, index=pd.RangeIndex(0, 4))
        table = pa.Table.from_pandas(df)
        table_no_index_name = pa.Table.from_pandas(df2)
        assert len(table.schema) == 1
        result = table.to_pandas()
        tm.assert_frame_equal(result, df)
        assert isinstance(result.index, pd.RangeIndex)
        assert _pandas_api.get_rangeindex_attribute(result.index, 'step') == 2
        assert result.index.name == index_name
        result2 = table_no_index_name.to_pandas()
        tm.assert_frame_equal(result2, df2)
        assert isinstance(result2.index, pd.RangeIndex)
        assert _pandas_api.get_rangeindex_attribute(result2.index, 'step') == 1
        assert result2.index.name is None

    def test_range_index_force_serialization(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4]}, index=pd.RangeIndex(0, 8, step=2, name='foo'))
        table = pa.Table.from_pandas(df, preserve_index=True)
        assert table.num_columns == 2
        assert 'foo' in table.column_names
        restored = table.to_pandas()
        tm.assert_frame_equal(restored, df)

    def test_rangeindex_doesnt_warn(self):
        df = pd.DataFrame(np.random.randn(4, 2), columns=['a', 'b'])
        with warnings.catch_warnings():
            warnings.simplefilter(action='error')
            warnings.filterwarnings('ignore', 'make_block is deprecated', DeprecationWarning)
            _check_pandas_roundtrip(df, preserve_index=True)

    def test_multiindex_columns(self):
        columns = pd.MultiIndex.from_arrays([['one', 'two'], ['X', 'Y']])
        df = pd.DataFrame([(1, 'a'), (2, 'b'), (3, 'c')], columns=columns)
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_multiindex_columns_with_dtypes(self):
        columns = pd.MultiIndex.from_arrays([['one', 'two'], pd.DatetimeIndex(['2017-08-01', '2017-08-02'])], names=['level_1', 'level_2'])
        df = pd.DataFrame([(1, 'a'), (2, 'b'), (3, 'c')], columns=columns)
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_multiindex_with_column_dtype_object(self):
        df = pd.DataFrame([1], columns=pd.Index([1], dtype=object))
        _check_pandas_roundtrip(df, preserve_index=True)
        df = pd.DataFrame([1], columns=pd.Index([1.1], dtype=object))
        _check_pandas_roundtrip(df, preserve_index=True)
        df = pd.DataFrame([1], columns=pd.Index([datetime(2018, 1, 1)], dtype='object'))
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_multiindex_columns_unicode(self):
        columns = pd.MultiIndex.from_arrays([['あ', 'い'], ['X', 'Y']])
        df = pd.DataFrame([(1, 'a'), (2, 'b'), (3, 'c')], columns=columns)
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_multiindex_doesnt_warn(self):
        columns = pd.MultiIndex.from_arrays([['one', 'two'], ['X', 'Y']])
        df = pd.DataFrame([(1, 'a'), (2, 'b'), (3, 'c')], columns=columns)
        with warnings.catch_warnings():
            warnings.simplefilter(action='error')
            warnings.filterwarnings('ignore', 'make_block is deprecated', DeprecationWarning)
            _check_pandas_roundtrip(df, preserve_index=True)

    def test_integer_index_column(self):
        df = pd.DataFrame([(1, 'a'), (2, 'b'), (3, 'c')])
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_index_metadata_field_name(self):
        df = pd.DataFrame([(1, 'a', 3.1), (2, 'b', 2.2), (3, 'c', 1.3)], index=pd.MultiIndex.from_arrays([['c', 'b', 'a'], [3, 2, 1]], names=[None, 'foo']), columns=['a', None, '__index_level_0__'])
        with pytest.warns(UserWarning):
            t = pa.Table.from_pandas(df, preserve_index=True)
        js = t.schema.pandas_metadata
        col1, col2, col3, idx0, foo = js['columns']
        assert col1['name'] == 'a'
        assert col1['name'] == col1['field_name']
        assert col2['name'] is None
        assert col2['field_name'] == 'None'
        assert col3['name'] == '__index_level_0__'
        assert col3['name'] == col3['field_name']
        idx0_descr, foo_descr = js['index_columns']
        assert idx0_descr == '__index_level_0__'
        assert idx0['field_name'] == idx0_descr
        assert idx0['name'] is None
        assert foo_descr == 'foo'
        assert foo['field_name'] == foo_descr
        assert foo['name'] == foo_descr

    def test_categorical_column_index(self):
        df = pd.DataFrame([(1, 'a', 2.0), (2, 'b', 3.0), (3, 'c', 4.0)], columns=pd.Index(list('def'), dtype='category'))
        t = pa.Table.from_pandas(df, preserve_index=True)
        js = t.schema.pandas_metadata
        column_indexes, = js['column_indexes']
        assert column_indexes['name'] is None
        assert column_indexes['pandas_type'] == 'categorical'
        assert column_indexes['numpy_type'] == 'int8'
        md = column_indexes['metadata']
        assert md['num_categories'] == 3
        assert md['ordered'] is False

    def test_string_column_index(self):
        df = pd.DataFrame([(1, 'a', 2.0), (2, 'b', 3.0), (3, 'c', 4.0)], columns=pd.Index(list('def'), name='stringz'))
        t = pa.Table.from_pandas(df, preserve_index=True)
        js = t.schema.pandas_metadata
        column_indexes, = js['column_indexes']
        assert column_indexes['name'] == 'stringz'
        assert column_indexes['name'] == column_indexes['field_name']
        assert column_indexes['numpy_type'] == 'object'
        assert column_indexes['pandas_type'] == 'unicode'
        md = column_indexes['metadata']
        assert len(md) == 1
        assert md['encoding'] == 'UTF-8'

    def test_datetimetz_column_index(self):
        df = pd.DataFrame([(1, 'a', 2.0), (2, 'b', 3.0), (3, 'c', 4.0)], columns=pd.date_range(start='2017-01-01', periods=3, tz='America/New_York'))
        t = pa.Table.from_pandas(df, preserve_index=True)
        js = t.schema.pandas_metadata
        column_indexes, = js['column_indexes']
        assert column_indexes['name'] is None
        assert column_indexes['pandas_type'] == 'datetimetz'
        assert column_indexes['numpy_type'] == 'datetime64[ns]'
        md = column_indexes['metadata']
        assert md['timezone'] == 'America/New_York'

    def test_datetimetz_row_index(self):
        df = pd.DataFrame({'a': pd.date_range(start='2017-01-01', periods=3, tz='America/New_York')})
        df = df.set_index('a')
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_categorical_row_index(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]})
        df['a'] = df.a.astype('category')
        df = df.set_index('a')
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_duplicate_column_names_does_not_crash(self):
        df = pd.DataFrame([(1, 'a'), (2, 'b')], columns=list('aa'))
        with pytest.raises(ValueError):
            pa.Table.from_pandas(df)

    def test_dictionary_indices_boundscheck(self):
        indices = [[0, 1], [0, -1]]
        for inds in indices:
            arr = pa.DictionaryArray.from_arrays(inds, ['a'], safe=False)
            batch = pa.RecordBatch.from_arrays([arr], ['foo'])
            table = pa.Table.from_batches([batch, batch, batch])
            with pytest.raises(IndexError):
                arr.to_pandas()
            with pytest.raises(IndexError):
                table.to_pandas()

    def test_unicode_with_unicode_column_and_index(self):
        df = pd.DataFrame({'あ': ['い']}, index=['う'])
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_mixed_column_names(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        for cols in [['あ', b'a'], [1, '2'], [1, 1.5]]:
            df.columns = pd.Index(cols, dtype=object)
            with pytest.warns(UserWarning):
                pa.Table.from_pandas(df)
            expected = df.copy()
            expected.columns = df.columns.values.astype(str)
            with pytest.warns(UserWarning):
                _check_pandas_roundtrip(df, expected=expected, preserve_index=True)

    def test_binary_column_name(self):
        if Version('2.0.0') <= Version(pd.__version__) < Version('2.3.0'):
            pytest.skip('Regression in pandas 2.0.0')
        column_data = ['い']
        key = 'あ'.encode()
        data = {key: column_data}
        df = pd.DataFrame(data)
        t = pa.Table.from_pandas(df, preserve_index=True)
        df2 = t.to_pandas()
        assert df.values[0] == df2.values[0]
        assert df.index.values[0] == df2.index.values[0]
        assert df.columns[0] == key

    def test_multiindex_duplicate_values(self):
        num_rows = 3
        numbers = list(range(num_rows))
        index = pd.MultiIndex.from_arrays([['foo', 'foo', 'bar'], numbers], names=['foobar', 'some_numbers'])
        df = pd.DataFrame({'numbers': numbers}, index=index)
        _check_pandas_roundtrip(df, preserve_index=True)

    def test_metadata_with_mixed_types(self):
        df = pd.DataFrame({'data': [b'some_bytes', 'some_unicode']})
        table = pa.Table.from_pandas(df)
        js = table.schema.pandas_metadata
        assert 'mixed' not in js
        data_column = js['columns'][0]
        assert data_column['pandas_type'] == 'bytes'
        assert data_column['numpy_type'] == 'object'

    def test_ignore_metadata(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['foo', 'bar', 'baz']}, index=['one', 'two', 'three'])
        table = pa.Table.from_pandas(df)
        result = table.to_pandas(ignore_metadata=True)
        expected = table.cast(table.schema.remove_metadata()).to_pandas()
        tm.assert_frame_equal(result, expected)

    def test_list_metadata(self):
        df = pd.DataFrame({'data': [[1], [2, 3, 4], [5] * 7]})
        schema = pa.schema([pa.field('data', type=pa.list_(pa.int64()))])
        table = pa.Table.from_pandas(df, schema=schema)
        js = table.schema.pandas_metadata
        assert 'mixed' not in js
        data_column = js['columns'][0]
        assert data_column['pandas_type'] == 'list[int64]'
        assert data_column['numpy_type'] == 'object'

    def test_struct_metadata(self):
        df = pd.DataFrame({'dicts': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]})
        table = pa.Table.from_pandas(df)
        pandas_metadata = table.schema.pandas_metadata
        assert pandas_metadata['columns'][0]['pandas_type'] == 'object'

    def test_decimal_metadata(self):
        expected = pd.DataFrame({'decimals': [decimal.Decimal('394092382910493.12341234678'), -decimal.Decimal('314292388910493.12343437128')]})
        table = pa.Table.from_pandas(expected)
        js = table.schema.pandas_metadata
        assert 'mixed' not in js
        data_column = js['columns'][0]
        assert data_column['pandas_type'] == 'decimal'
        assert data_column['numpy_type'] == 'object'
        assert data_column['metadata'] == {'precision': 26, 'scale': 11}

    def test_table_column_subset_metadata(self):
        for index in [pd.Index(['a', 'b', 'c'], name='index'), pd.date_range('2017-01-01', periods=3, tz='Europe/Brussels')]:
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]}, index=index)
            table = pa.Table.from_pandas(df)
            table_subset = table.remove_column(1)
            result = table_subset.to_pandas()
            expected = df[['a']]
            if isinstance(df.index, pd.DatetimeIndex):
                df.index.freq = None
            tm.assert_frame_equal(result, expected)
            table_subset2 = table_subset.remove_column(1)
            result = table_subset2.to_pandas()
            tm.assert_frame_equal(result, df[['a']].reset_index(drop=True))

    def test_to_pandas_column_subset_multiindex(self):
        df = pd.DataFrame({'first': list(range(5)), 'second': list(range(5)), 'value': np.arange(5)})
        table = pa.Table.from_pandas(df.set_index(['first', 'second']))
        subset = table.select(['first', 'value'])
        result = subset.to_pandas()
        expected = df[['first', 'value']].set_index('first')
        tm.assert_frame_equal(result, expected)

    def test_empty_list_metadata(self):
        c1 = [['test'], ['a', 'b'], None]
        c2 = [[], [], []]
        arrays = OrderedDict([('c1', pa.array(c1, type=pa.list_(pa.string()))), ('c2', pa.array(c2, type=pa.list_(pa.string())))])
        rb = pa.RecordBatch.from_arrays(list(arrays.values()), list(arrays.keys()))
        tbl = pa.Table.from_batches([rb])
        df = tbl.to_pandas()
        tbl2 = pa.Table.from_pandas(df)
        md2 = tbl2.schema.pandas_metadata
        df2 = tbl2.to_pandas()
        expected = pd.DataFrame(OrderedDict([('c1', c1), ('c2', c2)]))
        tm.assert_frame_equal(df2, expected)
        assert md2['columns'] == [{'name': 'c1', 'field_name': 'c1', 'metadata': None, 'numpy_type': 'object', 'pandas_type': 'list[unicode]'}, {'name': 'c2', 'field_name': 'c2', 'metadata': None, 'numpy_type': 'object', 'pandas_type': 'list[empty]'}]

    def test_metadata_pandas_version(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]})
        table = pa.Table.from_pandas(df)
        assert table.schema.pandas_metadata['pandas_version'] is not None

    def test_mismatch_metadata_schema(self):
        df = pd.DataFrame({'datetime': pd.date_range('2020-01-01', periods=3)})
        table = pa.Table.from_pandas(df)
        new_col = table['datetime'].cast(pa.timestamp('ns', tz='UTC'))
        new_table1 = table.set_column(0, pa.field('datetime', new_col.type), new_col)
        schema = pa.schema([('datetime', pa.timestamp('ns', tz='UTC'))])
        new_table2 = pa.Table.from_pandas(df, schema=schema)
        expected = df.copy()
        expected['datetime'] = expected['datetime'].dt.tz_localize('UTC')
        for new_table in [new_table1, new_table2]:
            assert new_table.schema.pandas_metadata is not None
            result = new_table.to_pandas()
            tm.assert_frame_equal(result, expected)