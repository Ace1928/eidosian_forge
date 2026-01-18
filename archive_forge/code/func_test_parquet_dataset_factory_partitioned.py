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
def test_parquet_dataset_factory_partitioned(tempdir):
    root_path = tempdir / 'test_parquet_dataset_factory_partitioned'
    metadata_path, table = _create_parquet_dataset_partitioned(root_path)
    partitioning = ds.partitioning(flavor='hive')
    dataset = ds.parquet_dataset(metadata_path, partitioning=partitioning)
    assert dataset.schema.equals(table.schema)
    assert len(dataset.files) == 2
    result = dataset.to_table()
    assert result.num_rows == 20
    result = result.to_pandas().sort_values('f1').reset_index(drop=True)
    expected = table.to_pandas()
    pd.testing.assert_frame_equal(result, expected)