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
@pytest.mark.parquet
@pytest.mark.pandas
def test_write_dataset_use_threads(tempdir):
    directory = tempdir / 'partitioned'
    _ = _create_parquet_dataset_partitioned(directory)
    dataset = ds.dataset(directory, partitioning='hive')
    partitioning = ds.partitioning(pa.schema([('part', pa.string())]), flavor='hive')
    target1 = tempdir / 'partitioned1'
    paths_written = []

    def file_visitor(written_file):
        paths_written.append(written_file.path)
    ds.write_dataset(dataset, target1, format='feather', partitioning=partitioning, use_threads=True, file_visitor=file_visitor)
    expected_paths = {target1 / 'part=a' / 'part-0.feather', target1 / 'part=b' / 'part-0.feather'}
    paths_written_set = set(map(pathlib.Path, paths_written))
    assert paths_written_set == expected_paths
    target2 = tempdir / 'partitioned2'
    ds.write_dataset(dataset, target2, format='feather', partitioning=partitioning, use_threads=False)
    result1 = ds.dataset(target1, format='feather', partitioning=partitioning)
    result2 = ds.dataset(target2, format='feather', partitioning=partitioning)
    assert result1.to_table().equals(result2.to_table())