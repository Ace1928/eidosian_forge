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
def test_parquet_dataset_factory_invalid(tempdir):
    root_path = tempdir / 'test_parquet_dataset_invalid'
    metadata_path, table = _create_parquet_dataset_simple(root_path)
    list(root_path.glob('*.parquet'))[0].unlink()
    dataset = ds.parquet_dataset(metadata_path)
    assert dataset.schema.equals(table.schema)
    assert len(dataset.files) == 4
    with pytest.raises(FileNotFoundError):
        dataset.to_table()