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
class TestConvertMisc:
    """
    Miscellaneous conversion tests.
    """
    type_pairs = [(np.int8, pa.int8()), (np.int16, pa.int16()), (np.int32, pa.int32()), (np.int64, pa.int64()), (np.uint8, pa.uint8()), (np.uint16, pa.uint16()), (np.uint32, pa.uint32()), (np.uint64, pa.uint64()), (np.float16, pa.float16()), (np.float32, pa.float32()), (np.float64, pa.float64()), (np.object_, pa.string()), (np.object_, pa.binary()), (np.object_, pa.binary(10)), (np.object_, pa.list_(pa.int64()))]

    def test_all_none_objects(self):
        df = pd.DataFrame({'a': [None, None, None]})
        _check_pandas_roundtrip(df)

    def test_all_none_category(self):
        df = pd.DataFrame({'a': [None, None, None]})
        df['a'] = df['a'].astype('category')
        _check_pandas_roundtrip(df)

    def test_empty_arrays(self):
        for dtype, pa_type in self.type_pairs:
            arr = np.array([], dtype=dtype)
            _check_array_roundtrip(arr, type=pa_type)

    def test_non_threaded_conversion(self):
        _non_threaded_conversion()

    def test_threaded_conversion_multiprocess(self):
        pool = mp.Pool(2)
        try:
            pool.apply(_threaded_conversion)
        finally:
            pool.close()
            pool.join()

    def test_category(self):
        repeats = 5
        v1 = ['foo', None, 'bar', 'qux', np.nan]
        v2 = [4, 5, 6, 7, 8]
        v3 = [b'foo', None, b'bar', b'qux', np.nan]
        arrays = {'cat_strings': pd.Categorical(v1 * repeats), 'cat_strings_with_na': pd.Categorical(v1 * repeats, categories=['foo', 'bar']), 'cat_ints': pd.Categorical(v2 * repeats), 'cat_binary': pd.Categorical(v3 * repeats), 'cat_strings_ordered': pd.Categorical(v1 * repeats, categories=['bar', 'qux', 'foo'], ordered=True), 'ints': v2 * repeats, 'ints2': v2 * repeats, 'strings': v1 * repeats, 'strings2': v1 * repeats, 'strings3': v3 * repeats}
        df = pd.DataFrame(arrays)
        _check_pandas_roundtrip(df)
        for k in arrays:
            _check_array_roundtrip(arrays[k])

    def test_category_implicit_from_pandas(self):

        def _check(v):
            arr = pa.array(v)
            result = arr.to_pandas()
            tm.assert_series_equal(pd.Series(result), pd.Series(v))
        arrays = [pd.Categorical(['a', 'b', 'c'], categories=['a', 'b']), pd.Categorical(['a', 'b', 'c'], categories=['a', 'b'], ordered=True)]
        for arr in arrays:
            _check(arr)

    def test_empty_category(self):
        df = pd.DataFrame({'cat': pd.Categorical([])})
        _check_pandas_roundtrip(df)

    def test_category_zero_chunks(self):
        for pa_type, dtype in [(pa.string(), 'object'), (pa.int64(), 'int64')]:
            a = pa.chunked_array([], pa.dictionary(pa.int8(), pa_type))
            result = a.to_pandas()
            expected = pd.Categorical([], categories=np.array([], dtype=dtype))
            tm.assert_series_equal(pd.Series(result), pd.Series(expected))
            table = pa.table({'a': a})
            result = table.to_pandas()
            expected = pd.DataFrame({'a': expected})
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('data,error_type', [({'a': ['a', 1, 2.0]}, pa.ArrowTypeError), ({'a': ['a', 1, 2.0]}, pa.ArrowTypeError), ({'a': [1, True]}, pa.ArrowTypeError), ({'a': [True, 'a']}, pa.ArrowInvalid), ({'a': [1, 'a']}, pa.ArrowInvalid), ({'a': [1.0, 'a']}, pa.ArrowInvalid)])
    def test_mixed_types_fails(self, data, error_type):
        df = pd.DataFrame(data)
        msg = 'Conversion failed for column a with type object'
        with pytest.raises(error_type, match=msg):
            pa.Table.from_pandas(df)

    def test_strided_data_import(self):
        cases = []
        columns = ['a', 'b', 'c']
        N, K = (100, 3)
        random_numbers = np.random.randn(N, K).copy() * 100
        numeric_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8']
        for type_name in numeric_dtypes:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cases.append(random_numbers.astype(type_name))
        cases.append(np.array([random_ascii(10) for i in range(N * K)], dtype=object).reshape(N, K).copy())
        boolean_objects = np.array([True, False, True] * N, dtype=object).reshape(N, K).copy()
        boolean_objects[5] = None
        cases.append(boolean_objects)
        cases.append(np.arange('2016-01-01T00:00:00.001', N * K, dtype='datetime64[ms]').reshape(N, K).copy())
        strided_mask = (random_numbers > 0).astype(bool)[:, 0]
        for case in cases:
            df = pd.DataFrame(case, columns=columns)
            col = df['a']
            _check_pandas_roundtrip(df)
            _check_array_roundtrip(col)
            _check_array_roundtrip(col, mask=strided_mask)

    def test_all_nones(self):

        def _check_series(s):
            converted = pa.array(s)
            assert isinstance(converted, pa.NullArray)
            assert len(converted) == 3
            assert converted.null_count == 3
            for item in converted:
                assert item is pa.NA
        _check_series(pd.Series([None] * 3, dtype=object))
        _check_series(pd.Series([np.nan] * 3, dtype=object))
        _check_series(pd.Series([None, np.nan, None], dtype=object))

    def test_partial_schema(self):
        data = OrderedDict([('a', [0, 1, 2, 3, 4]), ('b', np.array([-10, -5, 0, 5, 10], dtype=np.int32)), ('c', [-10, -5, 0, 5, 10])])
        df = pd.DataFrame(data)
        partial_schema = pa.schema([pa.field('c', pa.int64()), pa.field('a', pa.int64())])
        _check_pandas_roundtrip(df, schema=partial_schema, expected=df[['c', 'a']], expected_schema=partial_schema)

    def test_table_batch_empty_dataframe(self):
        df = pd.DataFrame({})
        _check_pandas_roundtrip(df, preserve_index=None)
        _check_pandas_roundtrip(df, preserve_index=None, as_batch=True)
        expected = pd.DataFrame(columns=pd.Index([]))
        _check_pandas_roundtrip(df, expected, preserve_index=False)
        _check_pandas_roundtrip(df, expected, preserve_index=False, as_batch=True)
        df2 = pd.DataFrame({}, index=[0, 1, 2])
        _check_pandas_roundtrip(df2, preserve_index=True)
        _check_pandas_roundtrip(df2, as_batch=True, preserve_index=True)

    def test_convert_empty_table(self):
        arr = pa.array([], type=pa.int64())
        empty_objects = pd.Series(np.array([], dtype=object))
        tm.assert_series_equal(arr.to_pandas(), pd.Series(np.array([], dtype=np.int64)))
        arr = pa.array([], type=pa.string())
        tm.assert_series_equal(arr.to_pandas(), empty_objects)
        arr = pa.array([], type=pa.list_(pa.int64()))
        tm.assert_series_equal(arr.to_pandas(), empty_objects)
        arr = pa.array([], type=pa.struct([pa.field('a', pa.int64())]))
        tm.assert_series_equal(arr.to_pandas(), empty_objects)

    def test_non_natural_stride(self):
        """
        ARROW-2172: converting from a Numpy array with a stride that's
        not a multiple of itemsize.
        """
        dtype = np.dtype([('x', np.int32), ('y', np.int16)])
        data = np.array([(42, -1), (-43, 2)], dtype=dtype)
        assert data.strides == (6,)
        arr = pa.array(data['x'], type=pa.int32())
        assert arr.to_pylist() == [42, -43]
        arr = pa.array(data['y'], type=pa.int16())
        assert arr.to_pylist() == [-1, 2]

    def test_array_from_strided_numpy_array(self):
        np_arr = np.arange(0, 10, dtype=np.float32)[1:-1:2]
        pa_arr = pa.array(np_arr, type=pa.float64())
        expected = pa.array([1.0, 3.0, 5.0, 7.0], type=pa.float64())
        pa_arr.equals(expected)

    def test_safe_unsafe_casts(self):
        df = pd.DataFrame({'A': list('abc'), 'B': np.linspace(0, 1, 3)})
        schema = pa.schema([pa.field('A', pa.string()), pa.field('B', pa.int32())])
        with pytest.raises(ValueError):
            pa.Table.from_pandas(df, schema=schema)
        table = pa.Table.from_pandas(df, schema=schema, safe=False)
        assert table.column('B').type == pa.int32()

    def test_error_sparse(self):
        try:
            df = pd.DataFrame({'a': pd.arrays.SparseArray([1, np.nan, 3])})
        except AttributeError:
            df = pd.DataFrame({'a': pd.SparseArray([1, np.nan, 3])})
        with pytest.raises(TypeError, match='Sparse pandas data'):
            pa.Table.from_pandas(df)