import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def write_and_read_back(self, data, mode='b'):
    b_data = bytes(data)
    with gzip.GzipFile(self.filename, 'w' + mode) as f:
        l = f.write(data)
    self.assertEqual(l, len(b_data))
    with gzip.GzipFile(self.filename, 'r' + mode) as f:
        self.assertEqual(f.read(), b_data)