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
def test_python_file_writelines(tmpdir):
    lines = [b'line1\n', b'line2\nline3']
    path = os.path.join(str(tmpdir), 'foo.txt')
    with open(path, 'wb') as f:
        try:
            f = pa.PythonFile(f, mode='w')
            assert f.writable()
            f.writelines(lines)
        finally:
            f.close()
    with open(path, 'rb') as f:
        try:
            f = pa.PythonFile(f, mode='r')
            assert f.readable()
            assert f.read() == b''.join(lines)
        finally:
            f.close()