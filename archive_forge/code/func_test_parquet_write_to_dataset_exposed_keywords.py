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
def test_parquet_write_to_dataset_exposed_keywords(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'partitioning'
    paths_written = []

    def file_visitor(written_file):
        paths_written.append(written_file.path)
    basename_template = 'part-{i}.parquet'
    pq.write_to_dataset(table, path, partitioning=['a'], file_visitor=file_visitor, basename_template=basename_template)
    expected_paths = {path / '1' / 'part-0.parquet', path / '2' / 'part-0.parquet', path / '3' / 'part-0.parquet'}
    paths_written_set = set(map(pathlib.Path, paths_written))
    assert paths_written_set == expected_paths