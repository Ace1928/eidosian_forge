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
def test_partitioning_pickling(pickle_module):
    schema = pa.schema([pa.field('i64', pa.int64()), pa.field('f64', pa.float64())])
    parts = [ds.DirectoryPartitioning(schema), ds.HivePartitioning(schema), ds.FilenamePartitioning(schema), ds.DirectoryPartitioning(schema, segment_encoding='none'), ds.FilenamePartitioning(schema, segment_encoding='none'), ds.HivePartitioning(schema, segment_encoding='none', null_fallback='xyz')]
    for part in parts:
        assert pickle_module.loads(pickle_module.dumps(part)) == part