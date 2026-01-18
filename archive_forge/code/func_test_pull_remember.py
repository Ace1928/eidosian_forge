import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_remember(self):
    """Pull changes from one branch to another and test parent location."""
    t = self.get_transport()
    tree_a = self.make_branch_and_tree('branch_a')
    branch_a = tree_a.branch
    self.build_tree(['branch_a/a'])
    tree_a.add('a')
    tree_a.commit('commit a')
    tree_b = branch_a.controldir.sprout('branch_b').open_workingtree()
    branch_b = tree_b.branch
    tree_c = branch_a.controldir.sprout('branch_c').open_workingtree()
    branch_c = tree_c.branch
    self.build_tree(['branch_a/b'])
    tree_a.add('b')
    tree_a.commit('commit b')
    parent = branch_b.get_parent()
    branch_b = branch.Branch.open('branch_b')
    branch_b.set_parent(None)
    self.assertEqual(None, branch_b.get_parent())
    out = self.run_bzr('pull', retcode=3, working_dir='branch_b')
    self.assertEqual(out, ('', 'brz: ERROR: No pull location known or specified.\n'))
    self.build_tree(['branch_b/d'])
    tree_b.add('d')
    tree_b.commit('commit d')
    out = self.run_bzr('pull ../branch_a', retcode=3, working_dir='branch_b')
    self.assertEqual(out, ('', 'brz: ERROR: These branches have diverged. Use the missing command to see how.\nUse the merge command to reconcile them.\n'))
    tree_b = tree_b.controldir.open_workingtree()
    branch_b = tree_b.branch
    self.assertEqual(parent, branch_b.get_parent())
    uncommit.uncommit(branch=branch_b, tree=tree_b)
    t.delete('branch_b/d')
    self.run_bzr('pull', working_dir='branch_b')
    branch_b = branch_b.controldir.open_branch()
    self.assertEqual(branch_b.get_parent(), parent)
    self.run_bzr('pull ../branch_c --remember', working_dir='branch_b')
    branch_b = branch_b.controldir.open_branch()
    self.assertEqual(branch_c.controldir.root_transport.base, branch_b.get_parent())