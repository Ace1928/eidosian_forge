import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_write_memoryview(self):
    data = memoryview(data1 * 50)
    self.write_and_read_back(data.tobytes())
    data = memoryview(bytes(range(256)))
    self.write_and_read_back(data.tobytes())