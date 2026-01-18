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
def test_py_filesystem():
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    assert isinstance(fs, PyFileSystem)
    assert fs.type_name == 'py::dummy'
    assert fs.handler is handler
    with pytest.raises(TypeError):
        PyFileSystem(None)