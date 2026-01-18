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
def test_open_input_stream_not_found(fs, pathfn):
    p = pathfn('open-input-stream-not-found')
    with pytest.raises(FileNotFoundError):
        fs.open_input_stream(p)