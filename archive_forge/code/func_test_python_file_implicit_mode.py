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
def test_python_file_implicit_mode(tmpdir):
    path = os.path.join(str(tmpdir), 'foo.txt')
    with open(path, 'wb') as f:
        pf = pa.PythonFile(f)
        assert pf.writable()
        assert not pf.readable()
        assert not pf.seekable()
        f.write(b'foobar\n')
    with open(path, 'rb') as f:
        pf = pa.PythonFile(f)
        assert pf.readable()
        assert not pf.writable()
        assert pf.seekable()
        assert pf.read() == b'foobar\n'
    bio = BytesIO()
    pf = pa.PythonFile(bio)
    assert pf.writable()
    assert not pf.readable()
    assert not pf.seekable()
    pf.write(b'foobar\n')
    assert bio.getvalue() == b'foobar\n'