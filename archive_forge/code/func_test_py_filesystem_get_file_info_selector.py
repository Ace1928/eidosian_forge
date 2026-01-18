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
def test_py_filesystem_get_file_info_selector():
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    selector = FileSelector(base_dir='somedir')
    infos = fs.get_file_info(selector)
    assert len(infos) == 2
    assert infos[0].path == 'somedir/file1'
    assert infos[0].type == FileType.File
    assert infos[0].size == 123
    assert infos[1].path == 'somedir/subdir1'
    assert infos[1].type == FileType.Directory
    assert infos[1].size is None
    selector = FileSelector(base_dir='somedir', recursive=True)
    infos = fs.get_file_info(selector)
    assert len(infos) == 3
    assert infos[0].path == 'somedir/file1'
    assert infos[1].path == 'somedir/subdir1'
    assert infos[2].path == 'somedir/subdir1/file2'
    selector = FileSelector(base_dir='notfound')
    with pytest.raises(FileNotFoundError):
        fs.get_file_info(selector)
    selector = FileSelector(base_dir='notfound', allow_not_found=True)
    assert fs.get_file_info(selector) == []