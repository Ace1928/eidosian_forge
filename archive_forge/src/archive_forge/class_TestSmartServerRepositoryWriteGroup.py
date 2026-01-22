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
class TestSmartServerRepositoryWriteGroup(tests.TestCaseWithMemoryTransport):

    def test_start_write_group(self):
        backing = self.get_transport()
        repo = self.make_repository('.')
        lock_token = repo.lock_write().repository_token
        self.addCleanup(repo.unlock)
        request_class = smart_repo.SmartServerRepositoryStartWriteGroup
        request = request_class(backing)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok', [])), request.execute(b'', lock_token))

    def test_start_write_group_unsuspendable(self):
        backing = self.get_transport()
        repo = self.make_repository('.', format='knit')
        lock_token = repo.lock_write().repository_token
        self.addCleanup(repo.unlock)
        request_class = smart_repo.SmartServerRepositoryStartWriteGroup
        request = request_class(backing)
        self.assertEqual(smart_req.FailedSmartServerResponse((b'UnsuspendableWriteGroup',)), request.execute(b'', lock_token))

    def test_commit_write_group(self):
        backing = self.get_transport()
        repo = self.make_repository('.')
        lock_token = repo.lock_write().repository_token
        self.addCleanup(repo.unlock)
        repo.start_write_group()
        tokens = repo.suspend_write_group()
        request_class = smart_repo.SmartServerRepositoryCommitWriteGroup
        request = request_class(backing)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok',)), request.execute(b'', lock_token, tokens))

    def test_abort_write_group(self):
        backing = self.get_transport()
        repo = self.make_repository('.')
        lock_token = repo.lock_write().repository_token
        repo.start_write_group()
        tokens = repo.suspend_write_group()
        self.addCleanup(repo.unlock)
        request_class = smart_repo.SmartServerRepositoryAbortWriteGroup
        request = request_class(backing)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok',)), request.execute(b'', lock_token, tokens))

    def test_check_write_group(self):
        backing = self.get_transport()
        repo = self.make_repository('.')
        lock_token = repo.lock_write().repository_token
        repo.start_write_group()
        tokens = repo.suspend_write_group()
        self.addCleanup(repo.unlock)
        request_class = smart_repo.SmartServerRepositoryCheckWriteGroup
        request = request_class(backing)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok',)), request.execute(b'', lock_token, tokens))

    def test_check_write_group_invalid(self):
        backing = self.get_transport()
        repo = self.make_repository('.')
        lock_token = repo.lock_write().repository_token
        self.addCleanup(repo.unlock)
        request_class = smart_repo.SmartServerRepositoryCheckWriteGroup
        request = request_class(backing)
        self.assertEqual(smart_req.FailedSmartServerResponse((b'UnresumableWriteGroup', [b'random'], b'Malformed write group token')), request.execute(b'', lock_token, [b'random']))