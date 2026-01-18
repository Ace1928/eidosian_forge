import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_quiet(self):
    self.make_branch_and_tree('.')
    out, err = self.run_bzr('update --quiet')
    self.assertEqual('', err)
    self.assertEqual('', out)