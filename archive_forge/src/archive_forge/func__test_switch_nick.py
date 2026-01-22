import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def _test_switch_nick(self, lightweight):
    """Check that the nick gets switched too."""
    tree1 = self.make_branch_and_tree('branch1')
    tree2 = self.make_branch_and_tree('branch2')
    tree2.pull(tree1.branch)
    checkout = tree1.branch.create_checkout('checkout', lightweight=lightweight)
    self.assertEqual(checkout.branch.nick, tree1.branch.nick)
    self.assertEqual(checkout.branch.get_config().has_explicit_nickname(), False)
    self.run_bzr('switch branch2', working_dir='checkout')
    checkout = WorkingTree.open('checkout')
    self.assertEqual(checkout.branch.nick, tree2.branch.nick)
    self.assertEqual(checkout.branch.get_config().has_explicit_nickname(), False)