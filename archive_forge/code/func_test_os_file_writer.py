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
def test_os_file_writer(tmpdir):
    SIZE = 4096
    arr = np.random.randint(0, 256, size=SIZE).astype('u1')
    data = arr.tobytes()[:SIZE]
    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(data)
    f2 = pa.OSFile(path, mode='w')
    f2.write(b'foo')
    with pa.OSFile(path) as f3:
        assert f3.size() == 3
    with pytest.raises(IOError):
        f2.read(5)
    f2.close()
    with pa.OSFile(path, mode='ab') as f4:
        f4.write(b'bar')
    with pa.OSFile(path) as f5:
        assert f5.size() == 6