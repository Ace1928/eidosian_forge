from breezy import errors
from breezy.tests import per_repository, test_server
from breezy.transport import memory
def test_start_write_group_read_locked_needs_write_lock(self):
    repo = self.make_repository('.')
    with repo.lock_read():
        self.assertRaises(errors.NotWriteLocked, repo.start_write_group)