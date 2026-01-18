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
def test_read_commit_from_file(self):
    sha = b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e'
    c = self.commit(sha)
    self.assertEqual(c.tree, tree_sha)
    self.assertEqual(c.parents, [b'0d89f20333fbb1d2f3a94da77f4981373d8f4310'])
    self.assertEqual(c.author, b'James Westby <jw+debian@jameswestby.net>')
    self.assertEqual(c.committer, b'James Westby <jw+debian@jameswestby.net>')
    self.assertEqual(c.commit_time, 1174759230)
    self.assertEqual(c.commit_timezone, 0)
    self.assertEqual(c.author_timezone, 0)
    self.assertEqual(c.message, b'Test commit\n')