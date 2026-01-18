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
def test_parquet_fragment_statistics_nulls(tempdir):
    table = pa.table({'a': [0, 1, None, None], 'b': ['a', 'b', None, None]})
    pq.write_table(table, tempdir / 'test.parquet', row_group_size=2)
    dataset = ds.dataset(tempdir / 'test.parquet', format='parquet')
    fragments = list(dataset.get_fragments())[0].split_by_row_group()
    assert fragments[1].row_groups[0].statistics == {}