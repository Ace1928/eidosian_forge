import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_zero_padded_file(self):
    with gzip.GzipFile(self.filename, 'wb') as f:
        f.write(data1 * 50)
    with open(self.filename, 'ab') as f:
        f.write(b'\x00' * 50)
    with gzip.GzipFile(self.filename, 'rb') as f:
        d = f.read()
        self.assertEqual(d, data1 * 50, 'Incorrect data in file')