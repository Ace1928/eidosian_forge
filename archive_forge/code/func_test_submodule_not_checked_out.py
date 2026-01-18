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