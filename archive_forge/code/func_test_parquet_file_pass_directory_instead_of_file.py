import io
import os
import sys
import pytest
import pyarrow as pa
def test_parquet_file_pass_directory_instead_of_file(tempdir):
    path = tempdir / 'directory'
    os.mkdir(str(path))
    msg = f"Cannot open for reading: path '{str(path)}' is a directory"
    with pytest.raises(IOError) as exc:
        pq.ParquetFile(path)
    if exc.errisinstance(PermissionError) and sys.platform == 'win32':
        return
    exc.match(msg)