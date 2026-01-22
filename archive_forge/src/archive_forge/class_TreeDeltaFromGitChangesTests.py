import os
import stat
from dulwich import __version__ as dulwich_version
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.index import IndexEntry, ConflictedIndexEntry
from dulwich.object_store import OverlayObjectStore
from dulwich.objects import S_IFGITLINK, ZERO_SHA, Blob, Tree
from ... import conflicts as _mod_conflicts
from ... import workingtree as _mod_workingtree
from ...bzr.inventorytree import InventoryTreeChange as TreeChange
from ...delta import TreeDelta
from ...tests import TestCase, TestCaseWithTransport
from ..mapping import default_mapping
from ..tree import tree_delta_from_git_changes
class TreeDeltaFromGitChangesTests(TestCase):

    def test_empty(self):
        delta = TreeDelta()
        changes = []
        self.assertEqual(delta, tree_delta_from_git_changes(changes, (default_mapping, default_mapping)))

    def test_missing(self):
        delta = TreeDelta()
        delta.removed.append(TreeChange(b'git:a', ('a', 'a'), False, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'a'), ('file', None), (True, False)))
        changes = [('remove', (b'a', stat.S_IFREG | 493, b'a' * 40), (b'a', 0, b'a' * 40))]
        self.assertEqual(delta, tree_delta_from_git_changes(changes, (default_mapping, default_mapping)))