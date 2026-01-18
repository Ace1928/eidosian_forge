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
def test_parquet_dataset_lazy_filtering(tempdir, open_logging_fs):
    fs, assert_opens = open_logging_fs
    root_path = tempdir / 'test_parquet_dataset_lazy_filtering'
    metadata_path, _ = _create_parquet_dataset_simple(root_path)
    with assert_opens([metadata_path]):
        dataset = ds.parquet_dataset(metadata_path, partitioning=ds.partitioning(flavor='hive'), filesystem=fs)
    with assert_opens([]):
        fragments = list(dataset.get_fragments())
    with assert_opens([]):
        list(dataset.get_fragments(ds.field('f1') > 15))
    with assert_opens([]):
        fragments[0].split_by_row_group(ds.field('f1') > 15)
    with assert_opens([]):
        rg_fragments = fragments[0].split_by_row_group()
        rg_fragments[0].ensure_complete_metadata()