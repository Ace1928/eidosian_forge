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
class GitWorkingTreeTests(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.tree = self.make_branch_and_tree('.', format='git')

    def test_conflict_list(self):
        self.assertIsInstance(self.tree.conflicts(), _mod_conflicts.ConflictList)

    def test_add_conflict(self):
        self.build_tree(['conflicted'])
        self.tree.add(['conflicted'])
        with self.tree.lock_tree_write():
            self.tree.index[b'conflicted'] = ConflictedIndexEntry(this=self.tree.index[b'conflicted'])
            self.tree._index_dirty = True
        conflicts = self.tree.conflicts()
        self.assertEqual(1, len(conflicts))

    def test_revert_empty(self):
        self.build_tree(['a'])
        self.tree.add(['a'])
        self.assertTrue(self.tree.is_versioned('a'))
        self.tree.revert(['a'])
        self.assertFalse(self.tree.is_versioned('a'))

    def test_is_ignored_directory(self):
        self.assertFalse(self.tree.is_ignored('a'))
        self.build_tree(['a/'])
        self.assertFalse(self.tree.is_ignored('a'))
        self.build_tree_contents([('.gitignore', 'a\n')])
        self.tree._ignoremanager = None
        self.assertTrue(self.tree.is_ignored('a'))
        self.build_tree_contents([('.gitignore', 'a/\n')])
        self.tree._ignoremanager = None
        self.assertTrue(self.tree.is_ignored('a'))

    def test_add_submodule_dir(self):
        subtree = self.make_branch_and_tree('asub', format='git')
        subtree.commit('Empty commit')
        self.tree.add(['asub'])
        with self.tree.lock_read():
            entry = self.tree.index[b'asub']
            self.assertEqual(entry.mode, S_IFGITLINK)
        self.assertEqual([], list(subtree.unknowns()))

    def test_add_submodule_file(self):
        os.mkdir('.git/modules')
        subbranch = self.make_branch('.git/modules/asub', format='git-bare')
        os.mkdir('asub')
        with open('asub/.git', 'w') as f:
            f.write('gitdir: ../.git/modules/asub\n')
        subtree = _mod_workingtree.WorkingTree.open('asub')
        subtree.commit('Empty commit')
        self.tree.add(['asub'])
        with self.tree.lock_read():
            entry = self.tree.index[b'asub']
            self.assertEqual(entry.mode, S_IFGITLINK)
        self.assertEqual([], list(subtree.unknowns()))