import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_revision(self):
    tree = self._create_sample_tree()
    checkout = tree.branch.create_checkout('checkout', lightweight=True)
    self.run_bzr(['switch', 'branch-1', '-r1'], working_dir='checkout')
    self.assertPathExists('checkout/file-1')
    self.assertPathDoesNotExist('checkout/file-2')