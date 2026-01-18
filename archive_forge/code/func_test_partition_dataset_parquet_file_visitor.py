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
def test_partition_dataset_parquet_file_visitor(tempdir):
    f1_vals = [item for chunk in range(4) for item in [chunk] * 10]
    f2_vals = [item * 10 for chunk in range(4) for item in [chunk] * 10]
    table = pa.table({'f1': f1_vals, 'f2': f2_vals, 'part': np.repeat(['a', 'b'], 20)})
    root_path = tempdir / 'partitioned'
    partitioning = ds.partitioning(pa.schema([('part', pa.string())]), flavor='hive')
    paths_written = []
    sample_metadata = None

    def file_visitor(written_file):
        nonlocal sample_metadata
        if written_file.metadata:
            sample_metadata = written_file.metadata
        paths_written.append(written_file.path)
    ds.write_dataset(table, root_path, format='parquet', partitioning=partitioning, use_threads=True, file_visitor=file_visitor)
    expected_paths = {root_path / 'part=a' / 'part-0.parquet', root_path / 'part=b' / 'part-0.parquet'}
    paths_written_set = set(map(pathlib.Path, paths_written))
    assert paths_written_set == expected_paths
    assert sample_metadata is not None
    assert sample_metadata.num_columns == 2