from breezy import errors, urlutils
from breezy.bzr import remote
from breezy.controldir import ControlDir
from breezy.tests import multiply_tests
from breezy.tests.per_repository import (TestCaseWithRepository,
def test_repository_format(self):
    tree = self.make_branch_and_tree('repo')
    repo = self.make_referring('referring', tree.branch.repository)
    self.assertIsInstance(repo._format, self.repository_format.__class__)