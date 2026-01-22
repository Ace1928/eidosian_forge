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
class ChangesBetweenGitTreeAndWorkingCopyTests(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.wt = self.make_branch_and_tree('.', format='git')
        self.store = self.wt.branch.repository._git.object_store

    def expectDelta(self, expected_changes, expected_extras=None, want_unversioned=False, tree_id=None, rename_detector=None):
        if tree_id is None:
            try:
                tree_id = self.store[self.wt.branch.repository._git.head()].tree
            except KeyError:
                tree_id = None
        with self.wt.lock_read():
            changes, extras = changes_between_git_tree_and_working_copy(self.store, tree_id, self.wt, want_unversioned=want_unversioned, rename_detector=rename_detector)
            self.assertEqual(expected_changes, list(changes))
        if expected_extras is None:
            expected_extras = set()
        self.assertEqual(set(expected_extras), set(extras))

    def test_empty(self):
        self.expectDelta([('add', (None, None, None), (b'', stat.S_IFDIR, Tree().id))])

    def test_added_file(self):
        self.build_tree(['a'])
        self.wt.add(['a'])
        a = Blob.from_string(b'contents of a\n')
        t = Tree()
        t.add(b'a', stat.S_IFREG | 420, a.id)
        self.expectDelta([('add', (None, None, None), (b'', stat.S_IFDIR, t.id)), ('add', (None, None, None), (b'a', stat.S_IFREG | 420, a.id))])

    def test_renamed_file(self):
        self.build_tree(['a'])
        self.wt.add(['a'])
        self.wt.rename_one('a', 'b')
        a = Blob.from_string(b'contents of a\n')
        self.store.add_object(a)
        oldt = Tree()
        oldt.add(b'a', stat.S_IFREG | 420, a.id)
        self.store.add_object(oldt)
        newt = Tree()
        newt.add(b'b', stat.S_IFREG | 420, a.id)
        self.store.add_object(newt)
        self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('delete', (b'a', stat.S_IFREG | 420, a.id), (None, None, None)), ('add', (None, None, None), (b'b', stat.S_IFREG | 420, a.id))], tree_id=oldt.id)
        if dulwich_version >= (0, 19, 15):
            self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('rename', (b'a', stat.S_IFREG | 420, a.id), (b'b', stat.S_IFREG | 420, a.id))], tree_id=oldt.id, rename_detector=RenameDetector(self.store))

    def test_copied_file(self):
        self.build_tree(['a'])
        self.wt.add(['a'])
        self.wt.copy_one('a', 'b')
        a = Blob.from_string(b'contents of a\n')
        self.store.add_object(a)
        oldt = Tree()
        oldt.add(b'a', stat.S_IFREG | 420, a.id)
        self.store.add_object(oldt)
        newt = Tree()
        newt.add(b'a', stat.S_IFREG | 420, a.id)
        newt.add(b'b', stat.S_IFREG | 420, a.id)
        self.store.add_object(newt)
        self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('add', (None, None, None), (b'b', stat.S_IFREG | 420, a.id))], tree_id=oldt.id)
        if dulwich_version >= (0, 19, 15):
            self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('copy', (b'a', stat.S_IFREG | 420, a.id), (b'b', stat.S_IFREG | 420, a.id))], tree_id=oldt.id, rename_detector=RenameDetector(self.store, find_copies_harder=True))
            self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('add', (None, None, None), (b'b', stat.S_IFREG | 420, a.id))], tree_id=oldt.id, rename_detector=RenameDetector(self.store, find_copies_harder=False))

    def test_added_unknown_file(self):
        self.build_tree(['a'])
        t = Tree()
        self.expectDelta([('add', (None, None, None), (b'', stat.S_IFDIR, t.id))])
        a = Blob.from_string(b'contents of a\n')
        t = Tree()
        t.add(b'a', stat.S_IFREG | 420, a.id)
        self.expectDelta([('add', (None, None, None), (b'', stat.S_IFDIR, t.id)), ('add', (None, None, None), (b'a', stat.S_IFREG | 420, a.id))], [b'a'], want_unversioned=True)

    def test_missing_added_file(self):
        self.build_tree(['a'])
        self.wt.add(['a'])
        os.unlink('a')
        a = Blob.from_string(b'contents of a\n')
        t = Tree()
        t.add(b'a', 0, ZERO_SHA)
        self.expectDelta([('add', (None, None, None), (b'', stat.S_IFDIR, t.id)), ('add', (None, None, None), (b'a', 0, ZERO_SHA))], [])

    def test_missing_versioned_file(self):
        self.build_tree(['a'])
        self.wt.add(['a'])
        self.wt.commit('')
        os.unlink('a')
        a = Blob.from_string(b'contents of a\n')
        oldt = Tree()
        oldt.add(b'a', stat.S_IFREG | 420, a.id)
        newt = Tree()
        newt.add(b'a', 0, ZERO_SHA)
        self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('modify', (b'a', stat.S_IFREG | 420, a.id), (b'a', 0, ZERO_SHA))])

    def test_versioned_replace_by_dir(self):
        self.build_tree(['a'])
        self.wt.add(['a'])
        self.wt.commit('')
        os.unlink('a')
        os.mkdir('a')
        olda = Blob.from_string(b'contents of a\n')
        oldt = Tree()
        oldt.add(b'a', stat.S_IFREG | 420, olda.id)
        newt = Tree()
        newa = Tree()
        newt.add(b'a', stat.S_IFDIR, newa.id)
        self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('modify', (b'a', stat.S_IFREG | 420, olda.id), (b'a', stat.S_IFDIR, newa.id))], want_unversioned=False)
        self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('modify', (b'a', stat.S_IFREG | 420, olda.id), (b'a', stat.S_IFDIR, newa.id))], want_unversioned=True)

    def test_extra(self):
        self.build_tree(['a'])
        newa = Blob.from_string(b'contents of a\n')
        newt = Tree()
        newt.add(b'a', stat.S_IFREG | 420, newa.id)
        self.expectDelta([('add', (None, None, None), (b'', stat.S_IFDIR, newt.id)), ('add', (None, None, None), (b'a', stat.S_IFREG | 420, newa.id))], [b'a'], want_unversioned=True)

    def test_submodule(self):
        self.subtree = self.make_branch_and_tree('a', format='git')
        a = Blob.from_string(b'irrelevant\n')
        self.build_tree_contents([('a/.git/HEAD', a.id)])
        with self.wt.lock_tree_write():
            index, index_path = self.wt._lookup_index(b'a')
            index[b'a'] = IndexEntry(0, 0, 0, 0, S_IFGITLINK, 0, 0, 0, a.id)
            self.wt._index_dirty = True
        t = Tree()
        t.add(b'a', S_IFGITLINK, a.id)
        self.store.add_object(t)
        self.expectDelta([], tree_id=t.id)

    def test_submodule_not_checked_out(self):
        a = Blob.from_string(b'irrelevant\n')
        with self.wt.lock_tree_write():
            index, index_path = self.wt._lookup_index(b'a')
            index[b'a'] = IndexEntry(0, 0, 0, 0, S_IFGITLINK, 0, 0, 0, a.id)
            self.wt._index_dirty = True
        os.mkdir(self.wt.abspath('a'))
        t = Tree()
        t.add(b'a', S_IFGITLINK, a.id)
        self.store.add_object(t)
        self.expectDelta([], tree_id=t.id)