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
def _create_directory_of_files(base_dir):
    table1 = pa.table({'a': range(9), 'b': [0.0] * 4 + [1.0] * 5})
    path1 = base_dir / 'test1.parquet'
    pq.write_table(table1, path1)
    table2 = pa.table({'a': range(9, 18), 'b': [0.0] * 4 + [1.0] * 5})
    path2 = base_dir / 'test2.parquet'
    pq.write_table(table2, path2)
    return ((table1, table2), (path1, path2))