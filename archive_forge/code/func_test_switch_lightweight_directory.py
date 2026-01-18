import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_lightweight_directory(self):
    """Test --directory option"""
    a_tree = self.make_branch_and_tree('a')
    self.build_tree_contents([('a/a', b'initial\n')])
    a_tree.add('a')
    a_tree.commit(message='initial')
    b_tree = a_tree.controldir.sprout('b').open_workingtree()
    self.build_tree_contents([('b/a', b'initial\nmore\n')])
    b_tree.commit(message='more')
    self.run_bzr('checkout --lightweight a checkout')
    self.run_bzr('switch --directory checkout b')
    self.assertFileEqual(b'initial\nmore\n', 'checkout/a')