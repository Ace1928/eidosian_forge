from breezy import errors, urlutils
from breezy.bzr import remote
from breezy.controldir import ControlDir
from breezy.tests import multiply_tests
from breezy.tests.per_repository import (TestCaseWithRepository,
class TestIncompatibleStacking(TestCaseWithRepository):

    def make_repo_and_incompatible_fallback(self):
        referring = self.make_repository('referring')
        if referring._format.supports_chks:
            different_fmt = '1.9'
        else:
            different_fmt = '2a'
        fallback = self.make_repository('fallback', format=different_fmt)
        return (referring, fallback)

    def test_add_fallback_repository_rejects_incompatible(self):
        referring, fallback = self.make_repo_and_incompatible_fallback()
        self.assertRaises(errors.IncompatibleRepositories, referring.add_fallback_repository, fallback)

    def test_add_fallback_doesnt_leave_fallback_locked(self):
        referring, fallback = self.make_repo_and_incompatible_fallback()
        self.addCleanup(referring.lock_read().unlock)
        self.assertFalse(fallback.is_locked())
        self.assertRaises(errors.IncompatibleRepositories, referring.add_fallback_repository, fallback)
        self.assertFalse(fallback.is_locked())