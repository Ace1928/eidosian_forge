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
class TestConvertStructTypes:
    """
    Conversion tests for struct types.
    """

    def test_pandas_roundtrip(self):
        df = pd.DataFrame({'dicts': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]})
        expected_schema = pa.schema([('dicts', pa.struct([('a', pa.int64()), ('b', pa.int64())]))])
        _check_pandas_roundtrip(df, expected_schema=expected_schema)
        _check_pandas_roundtrip(df, schema=expected_schema, expected_schema=expected_schema)

    def test_to_pandas(self):
        ints = pa.array([None, 2, 3], type=pa.int64())
        strs = pa.array(['a', None, 'c'], type=pa.string())
        bools = pa.array([True, False, None], type=pa.bool_())
        arr = pa.StructArray.from_arrays([ints, strs, bools], ['ints', 'strs', 'bools'])
        expected = pd.Series([{'ints': None, 'strs': 'a', 'bools': True}, {'ints': 2, 'strs': None, 'bools': False}, {'ints': 3, 'strs': 'c', 'bools': None}])
        series = pd.Series(arr.to_pandas())
        tm.assert_series_equal(series, expected)

    def test_to_pandas_multiple_chunks(self):
        gc.collect()
        bytes_start = pa.total_allocated_bytes()
        ints1 = pa.array([1], type=pa.int64())
        ints2 = pa.array([2], type=pa.int64())
        arr1 = pa.StructArray.from_arrays([ints1], ['ints'])
        arr2 = pa.StructArray.from_arrays([ints2], ['ints'])
        arr = pa.chunked_array([arr1, arr2])
        expected = pd.Series([{'ints': 1}, {'ints': 2}])
        series = pd.Series(arr.to_pandas())
        tm.assert_series_equal(series, expected)
        del series
        del arr
        del arr1
        del arr2
        del ints1
        del ints2
        bytes_end = pa.total_allocated_bytes()
        assert bytes_end == bytes_start

    def test_from_numpy(self):
        dt = np.dtype([('x', np.int32), (('y_title', 'y'), np.bool_)])
        ty = pa.struct([pa.field('x', pa.int32()), pa.field('y', pa.bool_())])
        data = np.array([], dtype=dt)
        arr = pa.array(data, type=ty)
        assert arr.to_pylist() == []
        data = np.array([(42, True), (43, False)], dtype=dt)
        arr = pa.array(data, type=ty)
        assert arr.to_pylist() == [{'x': 42, 'y': True}, {'x': 43, 'y': False}]
        arr = pa.array(data, mask=np.bool_([False, True]), type=ty)
        assert arr.to_pylist() == [{'x': 42, 'y': True}, None]
        dt = np.dtype([])
        ty = pa.struct([])
        data = np.array([], dtype=dt)
        arr = pa.array(data, type=ty)
        assert arr.to_pylist() == []
        data = np.array([(), ()], dtype=dt)
        arr = pa.array(data, type=ty)
        assert arr.to_pylist() == [{}, {}]

    def test_from_numpy_nested(self):
        dt = np.dtype([('x', np.dtype([('xx', np.int8), ('yy', np.bool_)])), ('y', np.int16), ('z', np.object_)])
        assert dt.itemsize == 12
        ty = pa.struct([pa.field('x', pa.struct([pa.field('xx', pa.int8()), pa.field('yy', pa.bool_())])), pa.field('y', pa.int16()), pa.field('z', pa.string())])
        data = np.array([], dtype=dt)
        arr = pa.array(data, type=ty)
        assert arr.to_pylist() == []
        data = np.array([((1, True), 2, 'foo'), ((3, False), 4, 'bar')], dtype=dt)
        arr = pa.array(data, type=ty)
        assert arr.to_pylist() == [{'x': {'xx': 1, 'yy': True}, 'y': 2, 'z': 'foo'}, {'x': {'xx': 3, 'yy': False}, 'y': 4, 'z': 'bar'}]

    @pytest.mark.slow
    @pytest.mark.large_memory
    def test_from_numpy_large(self):
        target_size = 3 * 1024 ** 3
        dt = np.dtype([('x', np.float64), ('y', 'object')])
        bs = 65536 - dt.itemsize
        block = b'.' * bs
        n = target_size // (bs + dt.itemsize)
        data = np.zeros(n, dtype=dt)
        data['x'] = np.random.random_sample(n)
        data['y'] = block
        data['x'][data['x'] < 0.2] = np.nan
        ty = pa.struct([pa.field('x', pa.float64()), pa.field('y', pa.binary())])
        arr = pa.array(data, type=ty, from_pandas=True)
        arr.validate(full=True)
        assert arr.num_chunks == 2

        def iter_chunked_array(arr):
            for chunk in arr.iterchunks():
                yield from chunk

        def check(arr, data, mask=None):
            assert len(arr) == len(data)
            xs = data['x']
            ys = data['y']
            for i, obj in enumerate(iter_chunked_array(arr)):
                try:
                    d = obj.as_py()
                    if mask is not None and mask[i]:
                        assert d is None
                    else:
                        x = xs[i]
                        if np.isnan(x):
                            assert d['x'] is None
                        else:
                            assert d['x'] == x
                        assert d['y'] == ys[i]
                except Exception:
                    print('Failed at index', i)
                    raise
        check(arr, data)
        del arr
        mask = np.random.random_sample(n) < 0.2
        arr = pa.array(data, type=ty, mask=mask, from_pandas=True)
        arr.validate(full=True)
        assert arr.num_chunks == 2
        check(arr, data, mask)
        del arr

    def test_from_numpy_bad_input(self):
        ty = pa.struct([pa.field('x', pa.int32()), pa.field('y', pa.bool_())])
        dt = np.dtype([('x', np.int32), ('z', np.bool_)])
        data = np.array([], dtype=dt)
        with pytest.raises(ValueError, match="Missing field 'y'"):
            pa.array(data, type=ty)
        data = np.int32([])
        with pytest.raises(TypeError, match='Expected struct array'):
            pa.array(data, type=ty)

    def test_from_tuples(self):
        df = pd.DataFrame({'tuples': [(1, 2), (3, 4)]})
        expected_df = pd.DataFrame({'tuples': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]})
        struct_type = pa.struct([('a', pa.int64()), ('b', pa.int64())])
        arr = np.asarray(df['tuples'])
        _check_array_roundtrip(arr, expected=expected_df['tuples'], type=struct_type)
        expected_schema = pa.schema([('tuples', struct_type)])
        _check_pandas_roundtrip(df, expected=expected_df, schema=expected_schema, expected_schema=expected_schema)

    def test_struct_of_dictionary(self):
        names = ['ints', 'strs']
        children = [pa.array([456, 789, 456]).dictionary_encode(), pa.array(['foo', 'foo', None]).dictionary_encode()]
        arr = pa.StructArray.from_arrays(children, names=names)
        rows_as_tuples = zip(*(child.to_pylist() for child in children))
        rows_as_dicts = [dict(zip(names, row)) for row in rows_as_tuples]
        expected = pd.Series(rows_as_dicts)
        tm.assert_series_equal(arr.to_pandas(), expected)
        arr = arr.take([0, None, 2])
        expected[1] = None
        tm.assert_series_equal(arr.to_pandas(), expected)