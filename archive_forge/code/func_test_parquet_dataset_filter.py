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
def test_parquet_dataset_filter(tempdir):
    root_path = tempdir / 'test_parquet_dataset_filter'
    metadata_path, _ = _create_parquet_dataset_simple(root_path)
    dataset = ds.parquet_dataset(metadata_path)
    result = dataset.to_table()
    assert result.num_rows == 40
    filtered_ds = dataset.filter(pc.field('f1') < 2)
    assert filtered_ds.to_table().num_rows == 20
    with pytest.raises(ValueError):
        filtered_ds.get_fragments()