import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def _test_shelve_symlink_target_change(self, link_name, old_target, new_target, shelve_change=False):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    os.symlink(old_target, link_name)
    tree.add(link_name, ids=b'foo-id')
    tree.commit('commit symlink')
    os.unlink(link_name)
    os.symlink(new_target, link_name)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('modify target', b'foo-id', link_name, old_target, new_target)], list(creator.iter_shelvable()))
    if shelve_change:
        creator.shelve_change(('modify target', b'foo-id', link_name, old_target, new_target))
    else:
        creator.shelve_modify_target(b'foo-id')
    creator.transform()
    self.assertEqual(old_target, osutils.readlink(link_name))
    s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
    limbo_name = creator.shelf_transform._limbo_name(s_trans_id)
    self.assertEqual(new_target, osutils.readlink(limbo_name))
    ptree = creator.shelf_transform.get_preview_tree()
    self.assertEqual(new_target, ptree.get_symlink_target(ptree.id2path(b'foo-id')))