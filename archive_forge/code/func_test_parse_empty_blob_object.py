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
def test_parse_empty_blob_object(self):
    sha = b'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
    b = self.get_blob(sha)
    self.assertEqual(b.data, b'')
    self.assertEqual(b.id, sha)
    self.assertEqual(b.sha().hexdigest().encode('ascii'), sha)