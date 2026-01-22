import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
class TestSmartServerBranchRequestUnlock(TestLockedBranch):

    def test_unlock_on_locked_branch_and_repo(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestUnlock(backing)
        branch = self.make_branch('.', format='knit')
        branch_token, repo_token = self.get_lock_tokens(branch)
        branch.leave_lock_in_place()
        branch.repository.leave_lock_in_place()
        branch.unlock()
        response = request.execute(b'', branch_token, repo_token)
        self.assertEqual(smart_req.SmartServerResponse((b'ok',)), response)
        new_branch = branch.controldir.open_branch()
        new_branch.lock_write()
        new_branch.unlock()

    def test_unlock_on_unlocked_branch_unlocked_repo(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestUnlock(backing)
        self.make_branch('.', format='knit')
        response = request.execute(b'', b'branch token', b'repo token')
        self.assertEqual(smart_req.SmartServerResponse((b'TokenMismatch',)), response)

    def test_unlock_on_unlocked_branch_locked_repo(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestUnlock(backing)
        branch = self.make_branch('.', format='knit')
        repo_token = branch.repository.lock_write().repository_token
        branch.repository.leave_lock_in_place()
        branch.repository.unlock()
        response = request.execute(b'', b'branch token', repo_token)
        self.assertEqual(smart_req.SmartServerResponse((b'TokenMismatch',)), response)
        branch.repository.lock_write(repo_token)
        branch.repository.dont_leave_lock_in_place()
        branch.repository.unlock()