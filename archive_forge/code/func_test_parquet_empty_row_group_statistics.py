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
@pytest.mark.pandas
@pytest.mark.parquet
def test_parquet_empty_row_group_statistics(tempdir):
    df = pd.DataFrame({'a': ['a', 'b', 'b'], 'b': [4, 5, 6]})[:0]
    df.to_parquet(tempdir / 'test.parquet', engine='pyarrow')
    dataset = ds.dataset(tempdir / 'test.parquet', format='parquet')
    fragments = list(dataset.get_fragments())[0].split_by_row_group()
    assert fragments[0].row_groups[0].statistics == {}