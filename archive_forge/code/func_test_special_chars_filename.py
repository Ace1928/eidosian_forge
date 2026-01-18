from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
def test_special_chars_filename(tempdir):
    table = pa.Table.from_arrays([pa.array([42])], ['ints'])
    filename = 'foo # bar'
    path = tempdir / filename
    assert not path.exists()
    _write_table(table, str(path))
    assert path.exists()
    table_read = _read_table(str(path))
    assert table_read.equals(table)