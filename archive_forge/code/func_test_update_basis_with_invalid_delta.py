import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_update_basis_with_invalid_delta(self):
    """When given an invalid delta, it should abort, and not be saved."""
    self.build_tree(['dir/', 'dir/file'])
    tree = self.create_wt4()
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.add(['dir', 'dir/file'], ids=[b'dir-id', b'file-id'])
    first_revision_id = tree.commit('init')
    root_id = tree.path2id('')
    state = tree.current_dirstate()
    state._read_dirblocks_if_needed()
    self.assertEqual([(b'', [((b'', b'', root_id), [b'd', b'd'])]), (b'', [((b'', b'dir', b'dir-id'), [b'd', b'd'])]), (b'dir', [((b'dir', b'file', b'file-id'), [b'f', b'f'])])], self.get_simple_dirblocks(state))
    tree.remove(['dir/file'])
    self.assertEqual([(b'', [((b'', b'', root_id), [b'd', b'd'])]), (b'', [((b'', b'dir', b'dir-id'), [b'd', b'd'])]), (b'dir', [((b'dir', b'file', b'file-id'), [b'a', b'f'])])], self.get_simple_dirblocks(state))
    tree.flush()
    new_dir = inventory.InventoryDirectory(b'dir-id', 'new-dir', root_id)
    new_dir.revision = b'new-revision-id'
    new_file = inventory.InventoryFile(b'file-id', 'new-file', root_id)
    new_file.revision = b'new-revision-id'
    self.assertRaises(errors.InconsistentDelta, tree.update_basis_by_delta, b'new-revision-id', [('dir', 'new-dir', b'dir-id', new_dir), ('dir/file', 'new-dir/new-file', b'file-id', new_file)])
    del state
    tree.unlock()
    tree.lock_read()
    self.assertEqual(first_revision_id, tree.last_revision())
    state = tree.current_dirstate()
    state._read_dirblocks_if_needed()
    self.assertEqual([(b'', [((b'', b'', root_id), [b'd', b'd'])]), (b'', [((b'', b'dir', b'dir-id'), [b'd', b'd'])]), (b'dir', [((b'dir', b'file', b'file-id'), [b'a', b'f'])])], self.get_simple_dirblocks(state))