import os
import zlib
from io import BytesIO
from tempfile import mkstemp
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_, assert_equal
from pytest import raises as assert_raises
from scipy.io.matlab._streams import (make_stream,
def test_seek_bad_checksum(self):
    data = np.random.randint(0, 256, 10).astype(np.uint8).tobytes()
    compressed_data = zlib.compress(data)
    compressed_data = compressed_data[:-1] + bytes([compressed_data[-1] + 1 & 255])
    compressed_stream = BytesIO(compressed_data)
    stream = ZlibInputStream(compressed_stream, len(compressed_data))
    assert_raises(zlib.error, stream.seek, len(data))