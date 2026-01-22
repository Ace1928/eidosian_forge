import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_ancestor(TestRevisionSpec):

    def test_non_exact_branch(self):
        self.assertRaises(errors.NotBranchError, self.get_in_history, 'ancestor:tree2/a')

    def test_simple(self):
        self.assertInHistoryIs(None, b'alt_r2', 'ancestor:tree2')
        tmp = self.tree
        self.tree = self.tree2
        self.tree2 = tmp
        self.assertInHistoryIs(2, b'alt_r2', 'ancestor:tree')

    def test_self(self):
        self.assertInHistoryIs(2, b'r2', 'ancestor:tree')

    def test_unrelated(self):
        new_tree = self.make_branch_and_tree('new_tree')
        new_tree.commit('Commit one', rev_id=b'new_r1')
        new_tree.commit('Commit two', rev_id=b'new_r2')
        new_tree.commit('Commit three', rev_id=b'new_r3')
        self.assertRaises(errors.NoCommonAncestor, self.get_in_history, 'ancestor:new_tree')

    def test_no_commits(self):
        new_tree = self.make_branch_and_tree('new_tree')
        self.assertRaises(errors.NoCommits, spec_in_history, 'ancestor:new_tree', self.tree.branch)
        self.assertRaises(errors.NoCommits, spec_in_history, 'ancestor:tree', new_tree.branch)

    def test_as_revision_id(self):
        self.assertAsRevisionId(b'alt_r2', 'ancestor:tree2')

    def test_default(self):
        self.assertRaises(errors.NotBranchError, self.get_in_history, 'ancestor:')
        tree3 = self.tree.controldir.sprout('tree3').open_workingtree()
        tree3.commit('foo', rev_id=b'r3')
        self.tree = tree3
        self.assertInHistoryIs(2, b'r2', 'ancestor:')