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
def test_subtree_filesystem():
    localfs = LocalFileSystem()
    subfs = SubTreeFileSystem('/base', localfs)
    assert subfs.base_path == '/base/'
    assert subfs.base_fs == localfs
    assert repr(subfs).startswith('SubTreeFileSystem(base_path=/base/, base_fs=<pyarrow._fs.LocalFileSystem')
    subfs = SubTreeFileSystem('/another/base/', LocalFileSystem())
    assert subfs.base_path == '/another/base/'
    assert subfs.base_fs == localfs
    assert repr(subfs).startswith('SubTreeFileSystem(base_path=/another/base/, base_fs=<pyarrow._fs.LocalFileSystem')