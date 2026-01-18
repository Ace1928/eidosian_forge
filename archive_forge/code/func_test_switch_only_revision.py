import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_only_revision(self):
    tree = self._create_sample_tree()
    checkout = tree.branch.create_checkout('checkout', lightweight=True)
    self.assertPathExists('checkout/file-1')
    self.assertPathExists('checkout/file-2')
    self.run_bzr(['switch', '-r1'], working_dir='checkout')
    self.assertPathExists('checkout/file-1')
    self.assertPathDoesNotExist('checkout/file-2')
    self.run_bzr_error(['brz switch --revision takes exactly one revision identifier'], ['switch', '-r0..2'], working_dir='checkout')