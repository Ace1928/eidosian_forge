import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_remember_sets_submit(self):
    tree_a = self.make_branch_and_tree('a')
    tree_a.commit('rev1')
    tree_b = tree_a.controldir.sprout('b').open_workingtree()
    self.assertIs(tree_b.branch.get_submit_branch(), None)
    out, err = self.run_bzr(['merge', '-d', 'b'])
    refreshed = workingtree.WorkingTree.open('b')
    self.assertIs(refreshed.branch.get_submit_branch(), None)
    out, err = self.run_bzr(['merge', '-d', 'b', 'a'])
    refreshed = workingtree.WorkingTree.open('b')
    self.assertEqual(refreshed.branch.get_submit_branch(), tree_a.controldir.root_transport.base)