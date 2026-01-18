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
def test_python_file_read_buffer():
    length = 10
    data = b'0123456798'
    dst_buf = bytearray(data)

    class DuckReader:

        def close(self):
            pass

        @property
        def closed(self):
            return False

        def read_buffer(self, nbytes):
            assert nbytes == length
            return memoryview(dst_buf)[:nbytes]
    duck_reader = DuckReader()
    with pa.PythonFile(duck_reader, mode='r') as f:
        buf = f.read_buffer(length)
        assert len(buf) == length
        assert memoryview(buf).tobytes() == dst_buf[:length]
        memoryview(buf)[0] = ord(b'x')
        assert dst_buf[0] == ord(b'x')