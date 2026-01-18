from breezy import errors
from breezy.tests import per_repository, test_server
from breezy.transport import memory
def test_start_write_group_write_locked_gets_None(self):
    repo = self.make_repository('.')
    repo.lock_write()
    self.assertEqual(None, repo.start_write_group())
    repo.commit_write_group()
    repo.unlock()