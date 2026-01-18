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
def test_open_dataset_from_fsspec(tempdir):
    table, path = _create_single_file(tempdir)
    fsspec = pytest.importorskip('fsspec')
    localfs = fsspec.filesystem('file')
    dataset = ds.dataset(path, filesystem=localfs)
    assert dataset.schema.equals(table.schema)