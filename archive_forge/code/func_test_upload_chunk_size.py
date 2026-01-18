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
def test_upload_chunk_size(self):
    test_data = b'bar' * 128
    bytes_file = io.BytesIO(test_data)
    self.fs.upload('foo', bytes_file, chunk_size=8)
    with self.fs.open('foo', 'rb') as f:
        data = f.read()
    self.assertEqual(data, test_data)