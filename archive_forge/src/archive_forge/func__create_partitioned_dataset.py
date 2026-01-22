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
def _create_partitioned_dataset(basedir):
    table = pa.table({'a': range(9), 'b': [0.0] * 4 + [1.0] * 5})
    path = basedir / 'dataset-partitioned'
    path.mkdir()
    for i in range(3):
        part = path / 'part={}'.format(i)
        part.mkdir()
        pq.write_table(table.slice(3 * i, 3), part / 'test.parquet')
    full_table = table.append_column('part', pa.array(np.repeat([0, 1, 2], 3), type=pa.int32()))
    return (full_table, path)