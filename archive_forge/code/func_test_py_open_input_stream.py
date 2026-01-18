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
def test_py_open_input_stream():
    fs = PyFileSystem(DummyHandler())
    with fs.open_input_stream('somefile') as f:
        assert f.read() == b'somefile:input_stream'
    with pytest.raises(FileNotFoundError):
        fs.open_input_stream('notfound')