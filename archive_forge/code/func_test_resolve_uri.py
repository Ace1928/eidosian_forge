import pyarrow as pa
from pyarrow import filesystem
import os
import pytest
def test_resolve_uri():
    uri = 'file:///home/user/myfile.parquet'
    fs, path = filesystem.resolve_filesystem_and_path(uri)
    assert isinstance(fs, filesystem.LocalFileSystem)
    assert path == '/home/user/myfile.parquet'