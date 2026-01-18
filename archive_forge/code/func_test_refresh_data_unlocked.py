from breezy import repository
from breezy.tests.per_repository import TestCaseWithRepository
def test_refresh_data_unlocked(self):
    repo = self.make_repository('.')
    repo.refresh_data()