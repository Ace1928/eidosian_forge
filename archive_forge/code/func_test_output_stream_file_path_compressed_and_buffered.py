import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
@pytest.mark.gzip
def test_output_stream_file_path_compressed_and_buffered(tmpdir):
    data = b'some test data\n' * 100 + b'eof\n'
    file_path = tmpdir / 'output_stream_compressed_and_buffered.gz'

    def check_data(file_path, data, **kwargs):
        with pa.output_stream(file_path, **kwargs) as stream:
            stream.write(data)
        with open(str(file_path), 'rb') as f:
            return f.read()
    result = check_data(file_path, data, buffer_size=32)
    assert gzip.decompress(result) == data
    result = check_data(file_path, data, buffer_size=1024)
    assert gzip.decompress(result) == data
    result = check_data(file_path, data, buffer_size=1024, compression='gzip')
    assert gzip.decompress(result) == data