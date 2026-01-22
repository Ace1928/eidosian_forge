import contextlib
from breezy import errors, tests, transform, transport
from breezy.bzr.workingtree_4 import (DirStateRevisionTree, WorkingTreeFormat4,
from breezy.git.tree import GitRevisionTree
from breezy.git.workingtree import GitWorkingTreeFormat
from breezy.revisiontree import RevisionTree
from breezy.tests import features
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.tests.per_workingtree import make_scenario as wt_make_scenario
from breezy.tests.per_workingtree import make_scenarios as wt_make_scenarios
from breezy.workingtree import format_registry
class TestTreeImplementationSupport(tests.TestCaseWithTransport):

    def test_revision_tree_from_workingtree_bzr(self):
        tree = self.make_branch_and_tree('.', format='bzr')
        tree = revision_tree_from_workingtree(self, tree)
        self.assertIsInstance(tree, RevisionTree)

    def test_revision_tree_from_workingtree(self):
        tree = self.make_branch_and_tree('.', format='git')
        tree = revision_tree_from_workingtree(self, tree)
        self.assertIsInstance(tree, GitRevisionTree)