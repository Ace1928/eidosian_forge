import os
from breezy import merge, tests, transform, workingtree
def test_preserve_execute(self):
    tree = self.tree_with_executable()
    tt = tree.transform()
    newfile = tt.trans_id_tree_path('newfile')
    tt.delete_contents(newfile)
    tt.create_file([b'Woooorld!'], newfile)
    tt.apply()
    tree = workingtree.WorkingTree.open('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.assertTrue(tree.is_executable('newfile'))
    transform.revert(tree, tree.basis_tree(), None, backups=True)
    with tree.get_file('newfile', 'rb') as f:
        self.assertEqual(b'helooo!', f.read())
    self.assertTrue(tree.is_executable('newfile'))