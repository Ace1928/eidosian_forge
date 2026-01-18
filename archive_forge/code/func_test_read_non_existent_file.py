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
def test_read_non_existent_file(tempdir):
    path = 'nonexistent-file.parquet'
    try:
        pq.read_table(path)
    except Exception as e:
        assert path in e.args[0]