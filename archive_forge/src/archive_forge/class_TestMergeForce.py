import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
class TestMergeForce(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.tree_a = self.make_branch_and_tree('a')
        self.build_tree(['a/foo'])
        self.tree_a.add(['foo'])
        self.tree_a.commit('add file')
        self.tree_b = self.tree_a.controldir.sprout('b').open_workingtree()
        self.build_tree_contents([('a/foo', b'change 1')])
        self.tree_a.commit('change file')
        self.tree_b.merge_from_branch(self.tree_a.branch)

    def test_merge_force(self):
        self.tree_a.commit('empty change to allow merge to run')
        self.run_bzr(['merge', '../a', '--force'], working_dir='b')

    def test_merge_with_uncommitted_changes(self):
        self.run_bzr_error(['Working tree .* has uncommitted changes'], ['merge', '../a'], working_dir='b')

    def test_merge_with_pending_merges(self):
        self.run_bzr(['revert', 'b'])
        self.run_bzr_error(['Working tree .* has uncommitted changes'], ['merge', '../a'], working_dir='b')