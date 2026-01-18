import os
from breezy import merge, tests, transform, workingtree
def test_revert_executable(self):
    tree = self.tree_with_executable()
    tt = tree.transform()
    newfile = tt.trans_id_tree_path('newfile')
    tt.set_executability(False, newfile)
    tt.apply()
    tree.lock_write()
    self.addCleanup(tree.unlock)
    transform.revert(tree, tree.basis_tree(), None)
    self.assertTrue(tree.is_executable('newfile'))