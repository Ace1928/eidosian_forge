import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_finds_relative_branch(self):
    """Switch will find 'foo' relative to the branch the checkout is of."""
    self.build_tree(['repo/'])
    tree1 = self.make_branch_and_tree('repo/brancha')
    tree1.commit('foo')
    tree2 = self.make_branch_and_tree('repo/branchb')
    tree2.pull(tree1.branch)
    branchb_id = tree2.commit('bar')
    checkout = tree1.branch.create_checkout('checkout', lightweight=True)
    self.run_bzr(['switch', 'branchb'], working_dir='checkout')
    self.assertEqual(branchb_id, checkout.last_revision())
    checkout = checkout.controldir.open_workingtree()
    self.assertEqual(tree2.branch.base, checkout.branch.base)