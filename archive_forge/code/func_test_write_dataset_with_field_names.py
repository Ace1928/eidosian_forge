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
def test_write_dataset_with_field_names(tempdir):
    table = pa.table({'a': ['x', 'y', None], 'b': ['x', 'y', 'z']})
    ds.write_dataset(table, tempdir, format='ipc', partitioning=['b'])
    load_back = ds.dataset(tempdir, format='ipc', partitioning=['b'])
    files = load_back.files
    partitioning_dirs = {str(pathlib.Path(f).relative_to(tempdir).parent) for f in files}
    assert partitioning_dirs == {'x', 'y', 'z'}
    load_back_table = load_back.to_table()
    assert load_back_table.equals(table)