import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
class TestRepositoryLocking(per_repository.TestCaseWithRepository):

    def test_leave_lock_in_place(self):
        repo = self.make_repository('r')
        token = repo.lock_write().repository_token
        try:
            if token is None:
                self.assertRaises(NotImplementedError, repo.leave_lock_in_place)
                return
            repo.leave_lock_in_place()
        finally:
            repo.unlock()
        self.assertRaises(errors.LockContention, repo.lock_write)
        repo.lock_write(token)
        repo.dont_leave_lock_in_place()
        repo.unlock()

    def test_dont_leave_lock_in_place(self):
        repo = self.make_repository('r')
        token = repo.lock_write().repository_token
        try:
            if token is None:
                self.assertRaises(NotImplementedError, repo.dont_leave_lock_in_place)
                return
            try:
                repo.leave_lock_in_place()
            except NotImplementedError:
                return
        finally:
            repo.unlock()
        new_repo = repo.controldir.open_repository()
        new_repo.lock_write(token=token)
        new_repo.dont_leave_lock_in_place()
        new_repo.unlock()
        repo.lock_write()
        repo.unlock()

    def test_lock_read_then_unlock(self):
        repo = self.make_repository('r')
        repo.lock_read()
        repo.unlock()

    def test_lock_read_returns_unlockable(self):
        repo = self.make_repository('r')
        self.assertThat(repo.lock_read, ReturnsUnlockable(repo))

    def test_lock_write_returns_unlockable(self):
        repo = self.make_repository('r')
        self.assertThat(repo.lock_write, ReturnsUnlockable(repo))