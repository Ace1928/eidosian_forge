import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def test_set_chunks(self):
    b = Blob()
    b.chunked = [b'te', b'st', b' 5\n']
    self.assertEqual(b'test 5\n', b.data)
    b.chunked = [b'te', b'st', b' 6\n']
    self.assertEqual(b'test 6\n', b.as_raw_string())
    self.assertEqual(b'test 6\n', bytes(b))