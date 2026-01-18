import os
from ...transform import ROOT_PARENT, conflict_pass, resolve_conflicts, revert
from . import TestCaseWithTransport
def test_revert_does_not_remove(self):
    tree = self.make_branch_and_tree('.', format='git')
    tt = tree.transform()
    dir1 = tt.new_directory('dir', ROOT_PARENT)
    tid = tt.new_file('name1', dir1, [b'content1'])
    tt.version_file(tid)
    tt.apply()
    tree.commit('start')
    with open('dir/name1', 'wb') as f:
        f.write(b'new content2')
    revert(tree, tree.basis_tree())
    self.assertEqual([], list(tree.iter_changes(tree.basis_tree())))