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
def test_parquet_dataset_factory(tempdir):
    root_path = tempdir / 'test_parquet_dataset'
    metadata_path, table = _create_parquet_dataset_simple(root_path)
    dataset = ds.parquet_dataset(metadata_path)
    assert dataset.schema.equals(table.schema)
    assert len(dataset.files) == 4
    result = dataset.to_table()
    assert result.num_rows == 40