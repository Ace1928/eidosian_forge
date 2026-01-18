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
def test_py_filesystem_pickling(pickle_module):
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    serialized = pickle_module.dumps(fs)
    restored = pickle_module.loads(serialized)
    assert isinstance(restored, FileSystem)
    assert restored == fs
    assert restored.handler == handler
    assert restored.type_name == 'py::dummy'