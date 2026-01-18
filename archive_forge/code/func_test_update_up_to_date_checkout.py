import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_up_to_date_checkout(self):
    self.make_branch_and_tree('branch')
    self.run_bzr('checkout branch checkout')
    sr = ScriptRunner()
    sr.run_script(self, '\n$ brz update checkout\n2>Tree is up to date at revision 0 of branch .../branch\n')