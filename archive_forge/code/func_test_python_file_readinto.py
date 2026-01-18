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
def test_python_file_readinto():
    length = 10
    data = b'some sample data longer than 10'
    dst_buf = bytearray(length)
    src_buf = BytesIO(data)
    with pa.PythonFile(src_buf, mode='r') as f:
        assert f.readinto(dst_buf) == 10
        assert dst_buf[:length] == data[:length]
        assert len(dst_buf) == length