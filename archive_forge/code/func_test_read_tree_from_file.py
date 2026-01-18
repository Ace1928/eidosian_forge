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
def test_read_tree_from_file(self):
    t = self.get_tree(tree_sha)
    self.assertEqual(t.items()[0], (b'a', 33188, a_sha))
    self.assertEqual(t.items()[1], (b'b', 33188, b_sha))