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
def test_open_input_file(fs, pathfn):
    p = pathfn('open-input-file')
    data = b'some data' * 1024
    with fs.open_output_stream(p) as s:
        s.write(data)
    read_from = len(b'some data') * 512
    with fs.open_input_file(p) as f:
        result = f.read()
    assert result == data
    with fs.open_input_file(p) as f:
        f.seek(read_from)
        result = f.read()
    assert result == data[read_from:]