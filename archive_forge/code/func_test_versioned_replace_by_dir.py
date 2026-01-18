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