from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
@pytest.mark.parametrize('table_type, dtype', [(ht.PyObjectHashTable, np.object_), (ht.Complex128HashTable, np.complex128), (ht.Int64HashTable, np.int64), (ht.UInt64HashTable, np.uint64), (ht.Float64HashTable, np.float64), (ht.Complex64HashTable, np.complex64), (ht.Int32HashTable, np.int32), (ht.UInt32HashTable, np.uint32), (ht.Float32HashTable, np.float32), (ht.Int16HashTable, np.int16), (ht.UInt16HashTable, np.uint16), (ht.Int8HashTable, np.int8), (ht.UInt8HashTable, np.uint8), (ht.IntpHashTable, np.intp)])
class TestHashTable:

    def test_get_set_contains_len(self, table_type, dtype):
        index = 5
        table = table_type(55)
        assert len(table) == 0
        assert index not in table
        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42
        table.set_item(index + 1, 41)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 42
        assert table.get_item(index + 1) == 41
        table.set_item(index, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 41
        assert index + 2 not in table
        table.set_item(index + 1, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 21
        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)

    def test_get_set_contains_len_mask(self, table_type, dtype):
        if table_type == ht.PyObjectHashTable:
            pytest.skip('Mask not supported for object')
        index = 5
        table = table_type(55, uses_mask=True)
        assert len(table) == 0
        assert index not in table
        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42
        with pytest.raises(KeyError, match='NA'):
            table.get_na()
        table.set_item(index + 1, 41)
        table.set_na(41)
        assert pd.NA in table
        assert index in table
        assert index + 1 in table
        assert len(table) == 3
        assert table.get_item(index) == 42
        assert table.get_item(index + 1) == 41
        assert table.get_na() == 41
        table.set_na(21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 3
        assert table.get_item(index + 1) == 41
        assert table.get_na() == 21
        assert index + 2 not in table
        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)

    def test_map_keys_to_values(self, table_type, dtype, writable):
        if table_type == ht.Int64HashTable:
            N = 77
            table = table_type()
            keys = np.arange(N).astype(dtype)
            vals = np.arange(N).astype(np.int64) + N
            keys.flags.writeable = writable
            vals.flags.writeable = writable
            table.map_keys_to_values(keys, vals)
            for i in range(N):
                assert table.get_item(keys[i]) == i + N

    def test_map_locations(self, table_type, dtype, writable):
        N = 8
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        for i in range(N):
            assert table.get_item(keys[i]) == i

    def test_map_locations_mask(self, table_type, dtype, writable):
        if table_type == ht.PyObjectHashTable:
            pytest.skip('Mask not supported for object')
        N = 3
        table = table_type(uses_mask=True)
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys, np.array([False, False, True]))
        for i in range(N - 1):
            assert table.get_item(keys[i]) == i
        with pytest.raises(KeyError, match=re.escape(str(keys[N - 1]))):
            table.get_item(keys[N - 1])
        assert table.get_na() == 2

    def test_lookup(self, table_type, dtype, writable):
        N = 3
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        result = table.lookup(keys)
        expected = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))

    def test_lookup_wrong(self, table_type, dtype):
        if dtype in (np.int8, np.uint8):
            N = 100
        else:
            N = 512
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        table.map_locations(keys)
        wrong_keys = np.arange(N).astype(dtype)
        result = table.lookup(wrong_keys)
        assert np.all(result == -1)

    def test_lookup_mask(self, table_type, dtype, writable):
        if table_type == ht.PyObjectHashTable:
            pytest.skip('Mask not supported for object')
        N = 3
        table = table_type(uses_mask=True)
        keys = (np.arange(N) + N).astype(dtype)
        mask = np.array([False, True, False])
        keys.flags.writeable = writable
        table.map_locations(keys, mask)
        result = table.lookup(keys, mask)
        expected = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))
        result = table.lookup(np.array([1 + N]).astype(dtype), np.array([False]))
        tm.assert_numpy_array_equal(result.astype(np.int64), np.array([-1], dtype=np.int64))

    def test_unique(self, table_type, dtype, writable):
        if dtype in (np.int8, np.uint8):
            N = 88
        else:
            N = 1000
        table = table_type()
        expected = (np.arange(N) + N).astype(dtype)
        keys = np.repeat(expected, 5)
        keys.flags.writeable = writable
        unique = table.unique(keys)
        tm.assert_numpy_array_equal(unique, expected)

    def test_tracemalloc_works(self, table_type, dtype):
        if dtype in (np.int8, np.uint8):
            N = 256
        else:
            N = 30000
        keys = np.arange(N).astype(dtype)
        with activated_tracemalloc():
            table = table_type()
            table.map_locations(keys)
            used = get_allocated_khash_memory()
            my_size = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_tracemalloc_for_empty(self, table_type, dtype):
        with activated_tracemalloc():
            table = table_type()
            used = get_allocated_khash_memory()
            my_size = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_get_state(self, table_type, dtype):
        table = table_type(1000)
        state = table.get_state()
        assert state['size'] == 0
        assert state['n_occupied'] == 0
        assert 'n_buckets' in state
        assert 'upper_bound' in state

    @pytest.mark.parametrize('N', range(1, 110))
    def test_no_reallocation(self, table_type, dtype, N):
        keys = np.arange(N).astype(dtype)
        preallocated_table = table_type(N)
        n_buckets_start = preallocated_table.get_state()['n_buckets']
        preallocated_table.map_locations(keys)
        n_buckets_end = preallocated_table.get_state()['n_buckets']
        assert n_buckets_start == n_buckets_end
        clean_table = table_type()
        clean_table.map_locations(keys)
        assert n_buckets_start == clean_table.get_state()['n_buckets']