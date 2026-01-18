import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_seek_write(self):
    with gzip.GzipFile(self.filename, 'w') as f:
        for pos in range(0, 256, 16):
            f.seek(pos)
            f.write(b'GZ\n')