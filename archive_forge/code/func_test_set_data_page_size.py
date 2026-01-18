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
def test_set_data_page_size():
    arr = pa.array([1, 2, 3] * 100000)
    t = pa.Table.from_arrays([arr], names=['f0'])
    page_sizes = [2 << 16, 2 << 18]
    for target_page_size in page_sizes:
        _check_roundtrip(t, data_page_size=target_page_size)