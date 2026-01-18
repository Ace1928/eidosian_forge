import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
def test_read_column_duplicated_selection(tempdir, version):
    table = pa.table([[1, 2, 3], [4, 5, 6], [7, 8, 9]], names=['a', 'b', 'c'])
    path = str(tempdir / 'data.feather')
    write_feather(table, path, version=version)
    expected = pa.table([[1, 2, 3], [4, 5, 6], [1, 2, 3]], names=['a', 'b', 'a'])
    for col_selection in [['a', 'b', 'a'], [0, 1, 0]]:
        result = read_table(path, columns=col_selection)
        assert result.equals(expected)