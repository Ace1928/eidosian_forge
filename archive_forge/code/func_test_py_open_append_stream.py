from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
def test_py_open_append_stream():
    fs = PyFileSystem(DummyHandler())
    with fs.open_append_stream('somefile') as f:
        f.write(b'data')