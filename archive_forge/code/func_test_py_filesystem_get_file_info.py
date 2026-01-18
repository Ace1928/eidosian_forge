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
def test_py_filesystem_get_file_info():
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    [info] = fs.get_file_info(['some/dir'])
    assert info.path == 'some/dir'
    assert info.type == FileType.Directory
    [info] = fs.get_file_info(['some/file'])
    assert info.path == 'some/file'
    assert info.type == FileType.File
    [info] = fs.get_file_info(['notfound'])
    assert info.path == 'notfound'
    assert info.type == FileType.NotFound
    with pytest.raises(TypeError):
        fs.get_file_info(['badtype'])
    with pytest.raises(IOError):
        fs.get_file_info(['xxx'])