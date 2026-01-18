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
def test_py_filesystem_lifetime():
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    assert isinstance(fs, PyFileSystem)
    wr = weakref.ref(handler)
    handler = None
    assert wr() is not None
    fs = None
    assert wr() is None
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    wr = weakref.ref(handler)
    handler = None
    assert wr() is fs.handler
    assert wr() is not None
    fs = None
    assert wr() is None