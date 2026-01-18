import os
from breezy import merge, tests, transform, workingtree
def tree_with_executable(self):
    tree = self.make_branch_and_tree('tree')
    tt = tree.transform()
    tt.new_file('newfile', tt.root, [b'helooo!'], b'newfile-id', True)
    tt.apply()
    with tree.lock_write():
        self.assertTrue(tree.is_executable('newfile'))
        tree.commit('added newfile')
    return tree