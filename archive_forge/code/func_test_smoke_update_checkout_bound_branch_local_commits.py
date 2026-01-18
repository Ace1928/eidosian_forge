import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_smoke_update_checkout_bound_branch_local_commits(self):
    master = self.make_branch_and_tree('master')
    master.commit('first commit')
    self.run_bzr('checkout master child')
    self.run_bzr('checkout --lightweight child checkout')
    wt = workingtree.WorkingTree.open('checkout')
    with open('master/file', 'w') as a_file:
        a_file.write('Foo')
    master.add(['file'])
    master_tip = master.commit('add file')
    with open('child/file_b', 'w') as a_file:
        a_file.write('Foo')
    child = workingtree.WorkingTree.open('child')
    child.add(['file_b'])
    child_tip = child.commit('add file_b', local=True)
    with open('checkout/file_c', 'w') as a_file:
        a_file.write('Foo')
    wt.add(['file_c'])
    out, err = self.run_bzr('update checkout')
    self.assertEqual('', out)
    self.assertEqualDiff("+N  file_b\nAll changes applied successfully.\n+N  file\nAll changes applied successfully.\nUpdated to revision 2 of branch %s\nYour local commits will now show as pending merges with 'brz status', and can be committed with 'brz commit'.\n" % osutils.pathjoin(self.test_dir, 'master'), err)
    self.assertEqual([master_tip, child_tip], wt.get_parent_ids())
    self.assertPathExists('checkout/file')
    self.assertPathExists('checkout/file_b')
    self.assertPathExists('checkout/file_c')
    self.assertTrue(wt.has_filename('file_c'))