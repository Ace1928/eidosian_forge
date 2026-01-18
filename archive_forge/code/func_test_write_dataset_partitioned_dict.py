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
def test_write_dataset_partitioned_dict(tempdir):
    directory = tempdir / 'partitioned'
    _ = _create_parquet_dataset_partitioned(directory)
    dataset = ds.dataset(directory, partitioning=ds.HivePartitioning.discover(infer_dictionary=True))
    target = tempdir / 'partitioned-dir-target'
    expected_paths = [target / 'a', target / 'a' / 'part-0.arrow', target / 'b', target / 'b' / 'part-0.arrow']
    partitioning = ds.partitioning(pa.schema([dataset.schema.field('part')]), dictionaries={'part': pa.array(['a', 'b'])})
    _check_dataset_roundtrip(dataset, str(target), expected_paths, 'f1', target, partitioning=partitioning)