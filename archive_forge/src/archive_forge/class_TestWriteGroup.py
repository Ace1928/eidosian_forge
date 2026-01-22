from breezy import errors
from breezy.tests import per_repository, test_server
from breezy.transport import memory
class TestWriteGroup(per_repository.TestCaseWithRepository):

    def test_start_write_group_unlocked_needs_write_lock(self):
        repo = self.make_repository('.')
        self.assertRaises(errors.NotWriteLocked, repo.start_write_group)

    def test_start_write_group_read_locked_needs_write_lock(self):
        repo = self.make_repository('.')
        with repo.lock_read():
            self.assertRaises(errors.NotWriteLocked, repo.start_write_group)

    def test_start_write_group_write_locked_gets_None(self):
        repo = self.make_repository('.')
        repo.lock_write()
        self.assertEqual(None, repo.start_write_group())
        repo.commit_write_group()
        repo.unlock()

    def test_start_write_group_twice_errors(self):
        repo = self.make_repository('.')
        repo.lock_write()
        repo.start_write_group()
        try:
            self.assertRaises(errors.BzrError, repo.start_write_group)
        finally:
            repo.commit_write_group()
            repo.unlock()

    def test_commit_write_group_does_not_error(self):
        repo = self.make_repository('.')
        repo.lock_write()
        repo.start_write_group()
        repo.commit_write_group()
        repo.unlock()

    def test_unlock_in_write_group(self):
        repo = self.make_repository('.')
        repo.lock_write()
        repo.start_write_group()
        self.assertLogsError(errors.BzrError, repo.unlock)
        self.assertFalse(repo.is_locked())
        self.assertRaises(errors.BzrError, repo.commit_write_group)
        self.assertRaises(errors.BzrError, repo.unlock)

    def test_is_in_write_group(self):
        repo = self.make_repository('.')
        self.assertFalse(repo.is_in_write_group())
        repo.lock_write()
        repo.start_write_group()
        self.assertTrue(repo.is_in_write_group())
        repo.commit_write_group()
        self.assertFalse(repo.is_in_write_group())
        repo.start_write_group()
        self.assertTrue(repo.is_in_write_group())
        repo.abort_write_group()
        self.assertFalse(repo.is_in_write_group())
        repo.unlock()

    def test_abort_write_group_gets_None(self):
        repo = self.make_repository('.')
        repo.lock_write()
        repo.start_write_group()
        self.assertEqual(None, repo.abort_write_group())
        repo.unlock()

    def test_abort_write_group_does_not_raise_when_suppressed(self):
        if self.transport_server is test_server.LocalURLServer:
            self.transport_server = None
        self.vfs_transport_factory = memory.MemoryServer
        repo = self.make_repository('repo')
        token = repo.lock_write()
        self.addCleanup(repo.unlock)
        repo.start_write_group()
        t = self.get_transport('')
        t.rename('repo', 'foo')
        self.addCleanup(t.rename, 'foo', 'repo')
        self.assertEqual(None, repo.abort_write_group(suppress_errors=True))