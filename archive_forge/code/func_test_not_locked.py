from breezy.tests.per_repository import TestCaseWithRepository
def test_not_locked(self):
    repo = self.make_repository('.')
    self.assertFalse(repo.is_locked())