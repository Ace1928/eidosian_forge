import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_out_of_date_standalone_tree(self):
    self.make_branch_and_tree('branch')
    self.run_bzr('checkout --lightweight branch checkout')
    self.build_tree(['checkout/file'])
    self.run_bzr('add checkout/file')
    self.run_bzr('commit -m add-file checkout')
    out, err = self.run_bzr('update branch')
    self.assertEqual('', out)
    self.assertEqualDiff('+N  file\nAll changes applied successfully.\nUpdated to revision 1 of branch %s\n' % osutils.pathjoin(self.test_dir, 'branch'), err)
    self.assertPathExists('branch/file')