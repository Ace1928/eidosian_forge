import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_lightweight_after_branch_moved(self):
    self.prepare_lightweight_switch()
    self.run_bzr('switch --force ../branch1', working_dir='tree')
    branch_location = WorkingTree.open('tree').branch.base
    self.assertEndsWith(branch_location, 'branch1/')