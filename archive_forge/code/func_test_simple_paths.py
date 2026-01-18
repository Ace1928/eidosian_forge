import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_simple_paths(self):
    """Compare strings that act like normal string comparison"""
    self.assertCmpByDirs(-1, b'a', b'b')
    self.assertCmpByDirs(-1, b'aa', b'ab')
    self.assertCmpByDirs(-1, b'ab', b'bb')
    self.assertCmpByDirs(-1, b'aaa', b'aab')
    self.assertCmpByDirs(-1, b'aab', b'abb')
    self.assertCmpByDirs(-1, b'abb', b'bbb')
    self.assertCmpByDirs(-1, b'aaaa', b'aaab')
    self.assertCmpByDirs(-1, b'aaab', b'aabb')
    self.assertCmpByDirs(-1, b'aabb', b'abbb')
    self.assertCmpByDirs(-1, b'abbb', b'bbbb')
    self.assertCmpByDirs(-1, b'aaaaa', b'aaaab')
    self.assertCmpByDirs(-1, b'a/a', b'a/b')
    self.assertCmpByDirs(-1, b'a/b', b'b/b')
    self.assertCmpByDirs(-1, b'a/a/a', b'a/a/b')
    self.assertCmpByDirs(-1, b'a/a/b', b'a/b/b')
    self.assertCmpByDirs(-1, b'a/b/b', b'b/b/b')
    self.assertCmpByDirs(-1, b'a/a/a/a', b'a/a/a/b')
    self.assertCmpByDirs(-1, b'a/a/a/b', b'a/a/b/b')
    self.assertCmpByDirs(-1, b'a/a/b/b', b'a/b/b/b')
    self.assertCmpByDirs(-1, b'a/b/b/b', b'b/b/b/b')
    self.assertCmpByDirs(-1, b'a/a/a/a/a', b'a/a/a/a/b')