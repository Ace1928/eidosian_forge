import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_up_to_date_light_checkout(self):
    self.make_branch_and_tree('branch')
    self.run_bzr('branch branch branch2')
    self.run_bzr('checkout --lightweight branch checkout')
    os.chdir('checkout')
    out, err = self.run_bzr('switch ../branch2')
    self.assertContainsRe(err, 'Tree is up to date at revision 0.\n')
    self.assertContainsRe(err, 'Switched to branch at .*/branch2.\n')
    self.assertEqual('', out)