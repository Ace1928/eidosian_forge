import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_paddedfile_getattr(self):
    self.test_write()
    with gzip.GzipFile(self.filename, 'rb') as f:
        self.assertTrue(hasattr(f.fileobj, 'name'))
        self.assertEqual(f.fileobj.name, self.filename)