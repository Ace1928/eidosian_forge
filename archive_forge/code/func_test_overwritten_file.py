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
def test_overwritten_file(version):
    path = random_path()
    TEST_FILES.append(path)
    num_values = 100
    np.random.seed(0)
    values = np.random.randint(0, 10, size=num_values)
    table = pa.table({'ints': values})
    write_feather(table, path)
    table = pa.table({'more_ints': values[0:num_values // 2]})
    _check_arrow_roundtrip(table, path=path)