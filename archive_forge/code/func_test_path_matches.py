from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_path_matches(self):
    walker = Walker(None, [], paths=[b'foo', b'bar', b'baz/quux'])
    self.assertTrue(walker._path_matches(b'foo'))
    self.assertTrue(walker._path_matches(b'foo/a'))
    self.assertTrue(walker._path_matches(b'foo/a/b'))
    self.assertTrue(walker._path_matches(b'bar'))
    self.assertTrue(walker._path_matches(b'baz/quux'))
    self.assertTrue(walker._path_matches(b'baz/quux/a'))
    self.assertFalse(walker._path_matches(None))
    self.assertFalse(walker._path_matches(b'oops'))
    self.assertFalse(walker._path_matches(b'fool'))
    self.assertFalse(walker._path_matches(b'baz'))
    self.assertFalse(walker._path_matches(b'baz/quu'))