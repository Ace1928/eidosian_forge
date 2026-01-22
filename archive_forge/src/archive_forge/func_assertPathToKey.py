from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def assertPathToKey(self, expected, path):
    self.assertEqual(expected, multiwalker.MultiWalker._path_to_key(path))