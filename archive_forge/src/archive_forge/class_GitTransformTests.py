import os
from ...transform import ROOT_PARENT, conflict_pass, resolve_conflicts, revert
from . import TestCaseWithTransport
class GitTransformTests(TestCaseWithTransport):

    def test_directory_exists(self):
        tree = self.make_branch_and_tree('.', format='git')
        tt = tree.transform()
        dir1 = tt.new_directory('dir', ROOT_PARENT)
        tt.new_file('name1', dir1, [b'content1'])
        dir2 = tt.new_directory('dir', ROOT_PARENT)
        tt.new_file('name2', dir2, [b'content2'])
        raw_conflicts = resolve_conflicts(tt, None, lambda t, c: conflict_pass(t, c))
        conflicts = tt.cook_conflicts(raw_conflicts)
        self.assertEqual([], list(conflicts))
        tt.apply()
        self.assertEqual({'name1', 'name2'}, set(os.listdir('dir')))

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