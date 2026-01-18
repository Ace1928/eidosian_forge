import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_write_incompatible_type(self):
    with gzip.GzipFile(self.filename, 'wb') as f:
        if six.PY2:
            with self.assertRaises(UnicodeEncodeError):
                f.write(u'ÿ')
        elif six.PY3:
            with self.assertRaises(TypeError):
                f.write(u'ÿ')
        with self.assertRaises(TypeError):
            f.write([1])
        f.write(data1)
    with gzip.GzipFile(self.filename, 'rb') as f:
        self.assertEqual(f.read(), data1)