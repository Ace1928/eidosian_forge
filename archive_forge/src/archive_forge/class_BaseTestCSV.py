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
class BaseTestCSV(abc.ABC):
    """Common tests which are shared by streaming and non streaming readers"""

    @abc.abstractmethod
    def read_bytes(self, b, **kwargs):
        """
        :param b: bytes to be parsed
        :param kwargs: arguments passed on to open the csv file
        :return: b parsed as a single RecordBatch
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def use_threads(self):
        """Whether this test is multi-threaded"""
        raise NotImplementedError

    @staticmethod
    def check_names(table, names):
        assert table.num_columns == len(names)
        assert table.column_names == names

    def test_header_skip_rows(self):
        rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
        opts = ReadOptions()
        opts.skip_rows = 1
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['ef', 'gh'])
        assert table.to_pydict() == {'ef': ['ij', 'mn'], 'gh': ['kl', 'op']}
        opts.skip_rows = 3
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['mn', 'op'])
        assert table.to_pydict() == {'mn': [], 'op': []}
        opts.skip_rows = 4
        with pytest.raises(pa.ArrowInvalid):
            table = self.read_bytes(rows, read_options=opts)
        rows = b'abcd\n,,,,,\nij,kl\nmn,op\n'
        opts.skip_rows = 2
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['ij', 'kl'])
        assert table.to_pydict() == {'ij': ['mn'], 'kl': ['op']}
        opts.skip_rows = 4
        opts.column_names = ['ij', 'kl']
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['ij', 'kl'])
        assert table.to_pydict() == {'ij': [], 'kl': []}

    def test_skip_rows_after_names(self):
        rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
        opts = ReadOptions()
        opts.skip_rows_after_names = 1
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['ab', 'cd'])
        assert table.to_pydict() == {'ab': ['ij', 'mn'], 'cd': ['kl', 'op']}
        opts.skip_rows_after_names = 3
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['ab', 'cd'])
        assert table.to_pydict() == {'ab': [], 'cd': []}
        opts.skip_rows_after_names = 4
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['ab', 'cd'])
        assert table.to_pydict() == {'ab': [], 'cd': []}
        rows = b'abcd\n,,,,,\nij,kl\nmn,op\n'
        opts.skip_rows_after_names = 2
        opts.column_names = ['f0', 'f1']
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['f0', 'f1'])
        assert table.to_pydict() == {'f0': ['ij', 'mn'], 'f1': ['kl', 'op']}
        opts = ReadOptions()
        rows = b'ab,cd\n"e\nf","g\n\nh"\n"ij","k\nl"\nmn,op'
        opts.skip_rows_after_names = 2
        parse_opts = ParseOptions()
        parse_opts.newlines_in_values = True
        table = self.read_bytes(rows, read_options=opts, parse_options=parse_opts)
        self.check_names(table, ['ab', 'cd'])
        assert table.to_pydict() == {'ab': ['mn'], 'cd': ['op']}
        opts.skip_rows_after_names = 2
        opts.block_size = 26
        table = self.read_bytes(rows, read_options=opts, parse_options=parse_opts)
        self.check_names(table, ['ab', 'cd'])
        assert table.to_pydict() == {'ab': ['mn'], 'cd': ['op']}
        opts = ReadOptions()
        rows, expected = make_random_csv(num_cols=5, num_rows=1000)
        opts.skip_rows_after_names = 900
        opts.block_size = len(rows) / 11
        table = self.read_bytes(rows, read_options=opts)
        assert table.schema == expected.schema
        assert table.num_rows == 100
        table_dict = table.to_pydict()
        for name, values in expected.to_pydict().items():
            assert values[900:] == table_dict[name]
        table = self.read_bytes(rows, read_options=opts, parse_options=parse_opts)
        assert table.schema == expected.schema
        assert table.num_rows == 100
        table_dict = table.to_pydict()
        for name, values in expected.to_pydict().items():
            assert values[900:] == table_dict[name]
        rows, expected = make_random_csv(num_cols=5, num_rows=200, write_names=False)
        opts = ReadOptions()
        opts.skip_rows = 37
        opts.skip_rows_after_names = 41
        opts.column_names = expected.schema.names
        table = self.read_bytes(rows, read_options=opts, parse_options=parse_opts)
        assert table.schema == expected.schema
        assert table.num_rows == expected.num_rows - opts.skip_rows - opts.skip_rows_after_names
        table_dict = table.to_pydict()
        for name, values in expected.to_pydict().items():
            assert values[opts.skip_rows + opts.skip_rows_after_names:] == table_dict[name]

    def test_row_number_offset_in_errors(self):

        def format_msg(msg_format, row, *args):
            if self.use_threads:
                row_info = ''
            else:
                row_info = 'Row #{}: '.format(row)
            return msg_format.format(row_info, *args)
        csv, _ = make_random_csv(4, 100, write_names=True)
        read_options = ReadOptions()
        read_options.block_size = len(csv) / 3
        convert_options = ConvertOptions()
        convert_options.column_types = {'a': pa.int32()}
        csv_bad_columns = csv + b'1,2\r\n'
        message_columns = format_msg('{}Expected 4 columns, got 2', 102)
        with pytest.raises(pa.ArrowInvalid, match=message_columns):
            self.read_bytes(csv_bad_columns, read_options=read_options, convert_options=convert_options)
        csv_bad_type = csv + b'a,b,c,d\r\n'
        message_value = format_msg("In CSV column #0: {}CSV conversion error to int32: invalid value 'a'", 102, csv)
        with pytest.raises(pa.ArrowInvalid, match=message_value):
            self.read_bytes(csv_bad_type, read_options=read_options, convert_options=convert_options)
        long_row = b'this is a long row' * 15 + b',3\r\n'
        csv_bad_columns_long = csv + long_row
        message_long = format_msg('{}Expected 4 columns, got 2: {} ...', 102, long_row[0:96].decode('utf-8'))
        with pytest.raises(pa.ArrowInvalid, match=message_long):
            self.read_bytes(csv_bad_columns_long, read_options=read_options, convert_options=convert_options)
        read_options.skip_rows_after_names = 47
        with pytest.raises(pa.ArrowInvalid, match=message_columns):
            self.read_bytes(csv_bad_columns, read_options=read_options, convert_options=convert_options)
        with pytest.raises(pa.ArrowInvalid, match=message_value):
            self.read_bytes(csv_bad_type, read_options=read_options, convert_options=convert_options)
        with pytest.raises(pa.ArrowInvalid, match=message_long):
            self.read_bytes(csv_bad_columns_long, read_options=read_options, convert_options=convert_options)
        read_options.skip_rows_after_names = 0
        csv, _ = make_random_csv(4, 100, write_names=False)
        read_options.column_names = ['a', 'b', 'c', 'd']
        csv_bad_columns = csv + b'1,2\r\n'
        message_columns = format_msg('{}Expected 4 columns, got 2', 101)
        with pytest.raises(pa.ArrowInvalid, match=message_columns):
            self.read_bytes(csv_bad_columns, read_options=read_options, convert_options=convert_options)
        csv_bad_columns_long = csv + long_row
        message_long = format_msg('{}Expected 4 columns, got 2: {} ...', 101, long_row[0:96].decode('utf-8'))
        with pytest.raises(pa.ArrowInvalid, match=message_long):
            self.read_bytes(csv_bad_columns_long, read_options=read_options, convert_options=convert_options)
        csv_bad_type = csv + b'a,b,c,d\r\n'
        message_value = format_msg("In CSV column #0: {}CSV conversion error to int32: invalid value 'a'", 101)
        message_value = message_value.format(len(csv))
        with pytest.raises(pa.ArrowInvalid, match=message_value):
            self.read_bytes(csv_bad_type, read_options=read_options, convert_options=convert_options)
        read_options.skip_rows = 23
        with pytest.raises(pa.ArrowInvalid, match=message_columns):
            self.read_bytes(csv_bad_columns, read_options=read_options, convert_options=convert_options)
        with pytest.raises(pa.ArrowInvalid, match=message_value):
            self.read_bytes(csv_bad_type, read_options=read_options, convert_options=convert_options)

    def test_invalid_row_handler(self, pickle_module):
        rows = b'a,b\nc\nd,e\nf,g,h\ni,j\n'
        parse_opts = ParseOptions()
        with pytest.raises(ValueError, match='Expected 2 columns, got 1: c'):
            self.read_bytes(rows, parse_options=parse_opts)
        parse_opts.invalid_row_handler = InvalidRowHandler('skip')
        table = self.read_bytes(rows, parse_options=parse_opts)
        assert table.to_pydict() == {'a': ['d', 'i'], 'b': ['e', 'j']}

        def row_num(x):
            return None if self.use_threads else x
        expected_rows = [InvalidRow(2, 1, row_num(2), 'c'), InvalidRow(2, 3, row_num(4), 'f,g,h')]
        assert parse_opts.invalid_row_handler.rows == expected_rows
        parse_opts.invalid_row_handler = InvalidRowHandler('error')
        with pytest.raises(ValueError, match='Expected 2 columns, got 1: c'):
            self.read_bytes(rows, parse_options=parse_opts)
        expected_rows = [InvalidRow(2, 1, row_num(2), 'c')]
        assert parse_opts.invalid_row_handler.rows == expected_rows
        parse_opts.invalid_row_handler = InvalidRowHandler('skip')
        parse_opts = pickle_module.loads(pickle_module.dumps(parse_opts))
        table = self.read_bytes(rows, parse_options=parse_opts)
        assert table.to_pydict() == {'a': ['d', 'i'], 'b': ['e', 'j']}