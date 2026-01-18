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
def test_python_file_read_at():
    data = b'some sample data'
    buf = BytesIO(data)
    f = pa.PythonFile(buf, mode='r')
    v = f.read_at(nbytes=5, offset=3)
    assert v == b'e sam'
    assert len(v) == 5
    w = f.read_at(nbytes=50, offset=0)
    assert w == data
    assert len(w) == 16