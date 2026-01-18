import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.pandas
@pytest.mark.parquet
def test_parquet_fragment_statistics(tempdir):
    table, dataset = _create_dataset_all_types(tempdir)
    fragment = list(dataset.get_fragments())[0]
    import datetime

    def dt_s(x):
        return datetime.datetime(1970, 1, 1, 0, 0, x)

    def dt_ms(x):
        return datetime.datetime(1970, 1, 1, 0, 0, 0, x * 1000)

    def dt_us(x):
        return datetime.datetime(1970, 1, 1, 0, 0, 0, x)
    date = datetime.date
    time = datetime.time
    row_group_fragments = list(fragment.split_by_row_group())
    assert row_group_fragments[0].row_groups is not None
    row_group = row_group_fragments[0].row_groups[0]
    assert row_group.num_rows == 3
    assert row_group.total_byte_size > 1000
    assert row_group.statistics == {'boolean': {'min': False, 'max': True}, 'int8': {'min': 1, 'max': 42}, 'uint8': {'min': 1, 'max': 42}, 'int16': {'min': 1, 'max': 42}, 'uint16': {'min': 1, 'max': 42}, 'int32': {'min': 1, 'max': 42}, 'uint32': {'min': 1, 'max': 42}, 'int64': {'min': 1, 'max': 42}, 'uint64': {'min': 1, 'max': 42}, 'float': {'min': 1.0, 'max': 42.0}, 'double': {'min': 1.0, 'max': 42.0}, 'utf8': {'min': 'a', 'max': 'z'}, 'binary': {'min': b'a', 'max': b'z'}, 'ts[s]': {'min': dt_s(1), 'max': dt_s(42)}, 'ts[ms]': {'min': dt_ms(1), 'max': dt_ms(42)}, 'ts[us]': {'min': dt_us(1), 'max': dt_us(42)}, 'date32': {'min': date(1970, 1, 2), 'max': date(1970, 2, 12)}, 'date64': {'min': date(1970, 1, 1), 'max': date(1970, 2, 18)}, 'time32': {'min': time(0, 0, 1), 'max': time(0, 0, 42)}, 'time64': {'min': time(0, 0, 0, 1), 'max': time(0, 0, 0, 42)}}