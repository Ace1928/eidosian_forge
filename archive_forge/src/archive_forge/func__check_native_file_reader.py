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
def _check_native_file_reader(FACTORY, sample_data, allow_read_out_of_bounds=True):
    path, data = sample_data
    f = FACTORY(path, mode='r')
    assert f.read(10) == data[:10]
    assert f.read(0) == b''
    assert f.tell() == 10
    assert f.read() == data[10:]
    assert f.size() == len(data)
    f.seek(0)
    assert f.tell() == 0
    if allow_read_out_of_bounds:
        f.seek(len(data) + 1)
        assert f.tell() == len(data) + 1
        assert f.read(5) == b''
    assert f.seek(3) == 3
    assert f.seek(3, os.SEEK_CUR) == 6
    assert f.tell() == 6
    ex_length = len(data) - 2
    assert f.seek(-2, os.SEEK_END) == ex_length
    assert f.tell() == ex_length