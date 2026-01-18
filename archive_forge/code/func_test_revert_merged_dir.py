import os
from breezy import merge, tests, transform, workingtree
def test_revert_merged_dir(self):
    """Reverting a merge that adds a directory deletes the directory"""
    source_tree = self.make_branch_and_tree('source')
    source_tree.commit('empty tree')
    target_tree = source_tree.controldir.sprout('target').open_workingtree()
    self.build_tree(['source/dir/', 'source/dir/contents'])
    source_tree.add(['dir', 'dir/contents'], ids=[b'dir-id', b'contents-id'])
    source_tree.commit('added dir')
    target_tree.lock_write()
    self.addCleanup(target_tree.unlock)
    merge.merge_inner(target_tree.branch, source_tree.basis_tree(), target_tree.basis_tree(), this_tree=target_tree)
    self.assertPathExists('target/dir')
    self.assertPathExists('target/dir/contents')
    target_tree.revert()
    self.assertPathDoesNotExist('target/dir/contents')
    self.assertPathDoesNotExist('target/dir')