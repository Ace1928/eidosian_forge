from __future__ import absolute_import, unicode_literals
import io
import itertools
import json
import os
import six
import time
import unittest
import warnings
from datetime import datetime
from six import text_type
import fs.copy
import fs.move
from fs import ResourceType, Seek, errors, glob, walk
from fs.opener import open_fs
from fs.subfs import ClosingSubFS, SubFS
def test_writebytes(self):
    all_bytes = b''.join((six.int2byte(n) for n in range(256)))
    self.fs.writebytes('foo', all_bytes)
    with self.fs.open('foo', 'rb') as f:
        _bytes = f.read()
    self.assertIsInstance(_bytes, bytes)
    self.assertEqual(_bytes, all_bytes)
    self.assert_bytes('foo', all_bytes)
    with self.assertRaises(TypeError):
        self.fs.writebytes('notbytes', 'unicode')