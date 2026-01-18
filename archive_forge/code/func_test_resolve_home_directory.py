import pyarrow as pa
from pyarrow import filesystem
import os
import pytest
@pytest.mark.filterwarnings('ignore:pyarrow.filesystem.LocalFileSystem')
def test_resolve_home_directory():
    uri = '~/myfile.parquet'
    fs, path = filesystem.resolve_filesystem_and_path(uri)
    assert isinstance(fs, filesystem.LocalFileSystem)
    assert path == os.path.expanduser(uri)
    local_fs = filesystem.LocalFileSystem()
    fs, path = filesystem.resolve_filesystem_and_path(uri, local_fs)
    assert path == os.path.expanduser(uri)