from breezy import errors, tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
class TestCommitStackedFailsAppropriately(TestCaseWithStackedTarget):

    def test_stacked_commit_fails_on_old_formats(self):
        base_tree, stacked_tree = self.make_stacked_target()
        format = stacked_tree.branch.repository._format
        if format.supports_chks:
            stacked_tree.commit('should succeed')
        else:
            self.assertRaises(errors.BzrError, stacked_tree.commit, 'unsupported format')