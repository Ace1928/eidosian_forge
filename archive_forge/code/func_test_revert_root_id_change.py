import os
from breezy import merge, tests, transform, workingtree
def test_revert_root_id_change(self):
    tree = self.make_branch_and_tree('.')
    tree.set_root_id(b'initial-root-id')
    self.build_tree(['file1'])
    tree.add(['file1'])
    tree.commit('first')
    tree.set_root_id(b'temp-root-id')
    self.assertEqual(b'temp-root-id', tree.path2id(''))
    tree.revert()
    self.assertEqual(b'initial-root-id', tree.path2id(''))