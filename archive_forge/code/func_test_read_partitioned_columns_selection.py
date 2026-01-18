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
@pytest.mark.pandas
def test_read_partitioned_columns_selection(tempdir):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir
    _partition_test_for_filesystem(fs, base_path)
    dataset = pq.ParquetDataset(base_path)
    result = dataset.read(columns=['values'])
    assert result.column_names == ['values']