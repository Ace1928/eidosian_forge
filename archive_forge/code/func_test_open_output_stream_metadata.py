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
def test_open_output_stream_metadata(fs, pathfn):
    p = pathfn('open-output-stream-metadata')
    metadata = {'Content-Type': 'x-pyarrow/test'}
    data = b'some data'
    with fs.open_output_stream(p, metadata=metadata) as f:
        f.write(data)
    with fs.open_input_stream(p) as f:
        assert f.read() == data
        got_metadata = f.metadata()
    if fs.type_name in ['s3', 'gcs'] or 'mock' in fs.type_name:
        for k, v in metadata.items():
            assert got_metadata[k] == v.encode()
    else:
        assert got_metadata == {}