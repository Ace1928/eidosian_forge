from breezy.tests.per_repository import TestCaseWithRepository
def test_read_locked(self):
    repo = self.make_repository('.')
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertTrue(repo.is_locked())