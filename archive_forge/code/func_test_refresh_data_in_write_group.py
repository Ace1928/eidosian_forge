from breezy import repository
from breezy.tests.per_repository import TestCaseWithRepository
def test_refresh_data_in_write_group(self):
    repo = self.make_repository('.')
    repo.lock_write()
    self.addCleanup(repo.unlock)
    repo.start_write_group()
    self.addCleanup(repo.abort_write_group)
    try:
        repo.refresh_data()
    except repository.IsInWriteGroupError:
        pass
    else:
        pass