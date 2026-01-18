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
def test_open_dataset_filesystem_fspath(tempdir):
    table, path = _create_single_file(tempdir)
    fspath = FSProtocolClass(path)
    dataset1 = ds.dataset(fspath)
    assert dataset1.schema.equals(table.schema)
    dataset2 = ds.dataset(fspath, filesystem=fs.LocalFileSystem())
    assert dataset2.schema.equals(table.schema)
    with pytest.raises(TypeError):
        ds.dataset(fspath, filesystem=fs._MockFileSystem())