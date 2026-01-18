from breezy import errors
from breezy.tests import per_repository, test_server
from breezy.transport import memory
def test_unlock_in_write_group(self):
    repo = self.make_repository('.')
    repo.lock_write()
    repo.start_write_group()
    self.assertLogsError(errors.BzrError, repo.unlock)
    self.assertFalse(repo.is_locked())
    self.assertRaises(errors.BzrError, repo.commit_write_group)
    self.assertRaises(errors.BzrError, repo.unlock)