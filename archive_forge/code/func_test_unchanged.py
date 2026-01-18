import stat
from dulwich.objects import Blob, Tree
from breezy.bzr.inventorytree import InventoryTreeChange as TreeChange
from breezy.delta import TreeDelta
from breezy.errors import PathsNotVersionedError
from breezy.git.mapping import default_mapping
from breezy.git.tree import (changes_from_git_changes,
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_unchanged(self):
    b = Blob.from_string(b'b')
    delta = self.transform([('unchanged', (b'a', stat.S_IFREG | 420, b), (b'a', stat.S_IFREG | 420, b))])
    expected_delta = TreeDelta()
    expected_delta.unchanged.append(TreeChange(b'git:a', ('a', 'a'), False, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'a'), ('file', 'file'), (False, False), False))