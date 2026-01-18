import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_read_with_extra(self):
    gzdata = b'\x1f\x8b\x08\x04\xb2\x17cQ\x02\xff\x05\x00Extra\x0bI-.\x01\x002\xd1Mx\x04\x00\x00\x00'
    with gzip.GzipFile(fileobj=io.BytesIO(gzdata)) as f:
        self.assertEqual(f.read(), b'Test')