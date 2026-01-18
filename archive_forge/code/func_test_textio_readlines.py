import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_textio_readlines(self):
    lines = (data1 * 50).decode('ascii').splitlines(True)
    self.test_write()
    with gzip.GzipFile(self.filename, 'r') as f:
        with io.TextIOWrapper(f, encoding='ascii') as t:
            self.assertEqual(t.readlines(), lines)