import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def test_read_table_duplicate_column_selection(tempdir):
    table = pa.table({'a': pa.array([1, 2, 3], pa.int32()), 'b': pa.array([1, 2, 3], pa.uint8())})
    pq.write_table(table, tempdir / 'data.parquet')
    result = pq.read_table(tempdir / 'data.parquet', columns=['a', 'a'])
    expected_schema = pa.schema([('a', 'int32'), ('a', 'int32')])
    assert result.column_names == ['a', 'a']
    assert result.schema == expected_schema