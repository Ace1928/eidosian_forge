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
def test_parquet_dataset_partitions_piece_path_with_fsspec(tempdir):
    fsspec = pytest.importorskip('fsspec')
    filesystem = fsspec.filesystem('file')
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, tempdir / 'data.parquet')
    path = str(tempdir).replace('\\', '/')
    dataset = pq.ParquetDataset(path, filesystem=filesystem)
    expected = path + '/data.parquet'
    assert dataset.fragments[0].path == expected