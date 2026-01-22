import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.csv import (
from pyarrow.tests import util
class BaseStreamingCSVRead(BaseTestCSV):

    def open_csv(self, csv, *args, **kwargs):
        """
        Reads the CSV file into memory using pyarrow's open_csv
        csv The CSV bytes
        args Positional arguments to be forwarded to pyarrow's open_csv
        kwargs Keyword arguments to be forwarded to pyarrow's open_csv
        """
        read_options = kwargs.setdefault('read_options', ReadOptions())
        read_options.use_threads = self.use_threads
        return open_csv(csv, *args, **kwargs)

    def open_bytes(self, b, **kwargs):
        return self.open_csv(pa.py_buffer(b), **kwargs)

    def check_reader(self, reader, expected_schema, expected_data):
        assert reader.schema == expected_schema
        batches = list(reader)
        assert len(batches) == len(expected_data)
        for batch, expected_batch in zip(batches, expected_data):
            batch.validate(full=True)
            assert batch.schema == expected_schema
            assert batch.to_pydict() == expected_batch

    def read_bytes(self, b, **kwargs):
        return self.open_bytes(b, **kwargs).read_all()

    def test_file_object(self):
        data = b'a,b\n1,2\n3,4\n'
        expected_data = {'a': [1, 3], 'b': [2, 4]}
        bio = io.BytesIO(data)
        reader = self.open_csv(bio)
        expected_schema = pa.schema([('a', pa.int64()), ('b', pa.int64())])
        self.check_reader(reader, expected_schema, [expected_data])

    def test_header(self):
        rows = b'abc,def,gh\n'
        reader = self.open_bytes(rows)
        expected_schema = pa.schema([('abc', pa.null()), ('def', pa.null()), ('gh', pa.null())])
        self.check_reader(reader, expected_schema, [])

    def test_inference(self):
        rows = b'a,b\n123,456\nabc,de\xff\ngh,ij\n'
        expected_schema = pa.schema([('a', pa.string()), ('b', pa.binary())])
        read_options = ReadOptions()
        read_options.block_size = len(rows)
        reader = self.open_bytes(rows, read_options=read_options)
        self.check_reader(reader, expected_schema, [{'a': ['123', 'abc', 'gh'], 'b': [b'456', b'de\xff', b'ij']}])
        read_options.block_size = len(rows) - 1
        reader = self.open_bytes(rows, read_options=read_options)
        self.check_reader(reader, expected_schema, [{'a': ['123', 'abc'], 'b': [b'456', b'de\xff']}, {'a': ['gh'], 'b': [b'ij']}])

    def test_inference_failure(self):
        rows = b'a,b\n123,456\nabc,de\xff\ngh,ij\n'
        read_options = ReadOptions()
        read_options.block_size = len(rows) - 7
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('a', pa.int64()), ('b', pa.int64())])
        assert reader.schema == expected_schema
        assert reader.read_next_batch().to_pydict() == {'a': [123], 'b': [456]}
        with pytest.raises(ValueError, match='CSV conversion error to int64'):
            reader.read_next_batch()
        with pytest.raises(StopIteration):
            reader.read_next_batch()

    def test_invalid_csv(self):
        rows = b'a,b\n1,2,3\n4,5\n6,7\n'
        read_options = ReadOptions()
        read_options.block_size = 10
        with pytest.raises(pa.ArrowInvalid, match='Expected 2 columns, got 3'):
            reader = self.open_bytes(rows, read_options=read_options)
        rows = b'a,b\n1,2\n3,4,5\n6,7\n'
        read_options.block_size = 8
        reader = self.open_bytes(rows, read_options=read_options)
        assert reader.read_next_batch().to_pydict() == {'a': [1], 'b': [2]}
        with pytest.raises(pa.ArrowInvalid, match='Expected 2 columns, got 3'):
            reader.read_next_batch()
        with pytest.raises(StopIteration):
            reader.read_next_batch()

    def test_options_delimiter(self):
        rows = b'a;b,c\nde,fg;eh\n'
        reader = self.open_bytes(rows)
        expected_schema = pa.schema([('a;b', pa.string()), ('c', pa.string())])
        self.check_reader(reader, expected_schema, [{'a;b': ['de'], 'c': ['fg;eh']}])
        opts = ParseOptions(delimiter=';')
        reader = self.open_bytes(rows, parse_options=opts)
        expected_schema = pa.schema([('a', pa.string()), ('b,c', pa.string())])
        self.check_reader(reader, expected_schema, [{'a': ['de,fg'], 'b,c': ['eh']}])

    def test_no_ending_newline(self):
        rows = b'a,b,c\n1,2,3\n4,5,6'
        reader = self.open_bytes(rows)
        expected_schema = pa.schema([('a', pa.int64()), ('b', pa.int64()), ('c', pa.int64())])
        self.check_reader(reader, expected_schema, [{'a': [1, 4], 'b': [2, 5], 'c': [3, 6]}])

    def test_empty_file(self):
        with pytest.raises(ValueError, match='Empty CSV file'):
            self.open_bytes(b'')

    def test_column_options(self):
        rows = b'1,2,3\n4,5,6'
        read_options = ReadOptions()
        read_options.column_names = ['d', 'e', 'f']
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('d', pa.int64()), ('e', pa.int64()), ('f', pa.int64())])
        self.check_reader(reader, expected_schema, [{'d': [1, 4], 'e': [2, 5], 'f': [3, 6]}])
        convert_options = ConvertOptions()
        convert_options.include_columns = ['f', 'e']
        reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
        expected_schema = pa.schema([('f', pa.int64()), ('e', pa.int64())])
        self.check_reader(reader, expected_schema, [{'e': [2, 5], 'f': [3, 6]}])
        convert_options.column_types = {'e': pa.string()}
        reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
        expected_schema = pa.schema([('f', pa.int64()), ('e', pa.string())])
        self.check_reader(reader, expected_schema, [{'e': ['2', '5'], 'f': [3, 6]}])
        convert_options.include_columns = ['g', 'f', 'e']
        with pytest.raises(KeyError, match="Column 'g' in include_columns does not exist"):
            reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
        convert_options.include_missing_columns = True
        reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
        expected_schema = pa.schema([('g', pa.null()), ('f', pa.int64()), ('e', pa.string())])
        self.check_reader(reader, expected_schema, [{'g': [None, None], 'e': ['2', '5'], 'f': [3, 6]}])
        convert_options.column_types = {'e': pa.string(), 'g': pa.float64()}
        reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
        expected_schema = pa.schema([('g', pa.float64()), ('f', pa.int64()), ('e', pa.string())])
        self.check_reader(reader, expected_schema, [{'g': [None, None], 'e': ['2', '5'], 'f': [3, 6]}])

    def test_encoding(self):
        rows = b'a,b\nun,\xe9l\xe9phant'
        read_options = ReadOptions()
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('a', pa.string()), ('b', pa.binary())])
        self.check_reader(reader, expected_schema, [{'a': ['un'], 'b': [b'\xe9l\xe9phant']}])
        read_options.encoding = 'latin1'
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('a', pa.string()), ('b', pa.string())])
        self.check_reader(reader, expected_schema, [{'a': ['un'], 'b': ['éléphant']}])
        rows = b'\xff\xfea\x00,\x00b\x00\n\x00u\x00n\x00,\x00\xe9\x00l\x00\xe9\x00p\x00h\x00a\x00n\x00t\x00'
        read_options.encoding = 'utf16'
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('a', pa.string()), ('b', pa.string())])
        self.check_reader(reader, expected_schema, [{'a': ['un'], 'b': ['éléphant']}])

    def test_small_random_csv(self):
        csv, expected = make_random_csv(num_cols=2, num_rows=10)
        reader = self.open_bytes(csv)
        table = reader.read_all()
        assert table.schema == expected.schema
        assert table.equals(expected)
        assert table.to_pydict() == expected.to_pydict()

    def test_stress_block_sizes(self):
        csv_base, expected = make_random_csv(num_cols=2, num_rows=500)
        block_sizes = [19, 21, 23, 26, 37, 111]
        csvs = [csv_base, csv_base.rstrip(b'\r\n')]
        for csv in csvs:
            for block_size in block_sizes:
                assert csv[:block_size].count(b'\n') >= 2
                read_options = ReadOptions(block_size=block_size)
                reader = self.open_bytes(csv, read_options=read_options)
                table = reader.read_all()
                assert table.schema == expected.schema
                if not table.equals(expected):
                    assert table.to_pydict() == expected.to_pydict()

    def test_batch_lifetime(self):
        gc.collect()
        old_allocated = pa.total_allocated_bytes()

        def check_one_batch(reader, expected):
            batch = reader.read_next_batch()
            assert batch.to_pydict() == expected
        rows = b'10,11\n12,13\n14,15\n16,17\n'
        read_options = ReadOptions()
        read_options.column_names = ['a', 'b']
        read_options.block_size = 6
        reader = self.open_bytes(rows, read_options=read_options)
        check_one_batch(reader, {'a': [10], 'b': [11]})
        allocated_after_first_batch = pa.total_allocated_bytes()
        check_one_batch(reader, {'a': [12], 'b': [13]})
        assert pa.total_allocated_bytes() <= allocated_after_first_batch
        check_one_batch(reader, {'a': [14], 'b': [15]})
        assert pa.total_allocated_bytes() <= allocated_after_first_batch
        check_one_batch(reader, {'a': [16], 'b': [17]})
        assert pa.total_allocated_bytes() <= allocated_after_first_batch
        with pytest.raises(StopIteration):
            reader.read_next_batch()
        assert pa.total_allocated_bytes() == old_allocated
        reader = None
        assert pa.total_allocated_bytes() == old_allocated

    def test_header_skip_rows(self):
        super().test_header_skip_rows()
        rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
        opts = ReadOptions()
        opts.skip_rows = 4
        opts.column_names = ['ab', 'cd']
        reader = self.open_bytes(rows, read_options=opts)
        with pytest.raises(StopIteration):
            assert reader.read_next_batch()

    def test_skip_rows_after_names(self):
        super().test_skip_rows_after_names()
        rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
        opts = ReadOptions()
        opts.skip_rows_after_names = 3
        reader = self.open_bytes(rows, read_options=opts)
        with pytest.raises(StopIteration):
            assert reader.read_next_batch()
        opts.skip_rows_after_names = 99999
        reader = self.open_bytes(rows, read_options=opts)
        with pytest.raises(StopIteration):
            assert reader.read_next_batch()