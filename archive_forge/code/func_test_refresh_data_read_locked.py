from breezy import repository
from breezy.tests.per_repository import TestCaseWithRepository
def test_refresh_data_read_locked(self):
    repo = self.make_repository('.')
    repo.lock_read()
    self.addCleanup(repo.unlock)
    repo.refresh_data()