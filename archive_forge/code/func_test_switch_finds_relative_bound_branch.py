import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_finds_relative_bound_branch(self):
    """Using switch on a heavy checkout should find master sibling

        The behaviour of lighweight and heavy checkouts should be
        consistent when using the convenient "switch to sibling" feature
        Both should switch to a sibling of the branch
        they are bound to, and not a sibling of themself"""
    self.build_tree(['repo/', 'heavyco/'])
    tree1 = self.make_branch_and_tree('repo/brancha')
    tree1.commit('foo')
    tree2 = self.make_branch_and_tree('repo/branchb')
    tree2.pull(tree1.branch)
    branchb_id = tree2.commit('bar')
    checkout = tree1.branch.create_checkout('heavyco/a', lightweight=False)
    self.run_bzr(['switch', 'branchb'], working_dir='heavyco/a')
    checkout = checkout.controldir.open_workingtree()
    self.assertEqual(branchb_id, checkout.last_revision())
    self.assertEqual(tree2.branch.base, checkout.branch.get_bound_location())