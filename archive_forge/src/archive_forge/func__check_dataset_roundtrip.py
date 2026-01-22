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
def _check_dataset_roundtrip(dataset, base_dir, expected_files, sort_col, base_dir_path=None, partitioning=None):
    base_dir_path = base_dir_path or base_dir
    ds.write_dataset(dataset, base_dir, format='arrow', partitioning=partitioning, use_threads=False)
    file_paths = list(base_dir_path.rglob('*'))
    assert set(file_paths) == set(expected_files)
    dataset2 = ds.dataset(base_dir_path, format='arrow', partitioning=partitioning)
    assert _sort_table(dataset2.to_table(), sort_col).equals(_sort_table(dataset.to_table(), sort_col))