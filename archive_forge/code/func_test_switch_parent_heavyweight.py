import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_parent_heavyweight(self):
    """Heavyweight checkout using brz switch."""
    bb, mb = self._checkout_and_switch()
    self.assertParent('repo/trunk', bb)
    self.assertParent('repo/trunk', mb)