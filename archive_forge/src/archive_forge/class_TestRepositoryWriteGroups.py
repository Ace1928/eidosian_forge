import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
class TestRepositoryWriteGroups(TestRemoteRepository):

    def test_start_write_group(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.lock_write', (b'quack/', b''), b'success', (b'ok', b'a token'))
        client.add_expected_call(b'Repository.start_write_group', (b'quack/', b'a token'), b'success', (b'ok', (b'token1',)))
        repo.lock_write()
        repo.start_write_group()

    def test_start_write_group_unsuspendable(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)

        def stub_ensure_real():
            client._calls.append(('_ensure_real',))
            repo._real_repository = _StubRealPackRepository(client._calls)
        repo._ensure_real = stub_ensure_real
        client.add_expected_call(b'Repository.lock_write', (b'quack/', b''), b'success', (b'ok', b'a token'))
        client.add_expected_call(b'Repository.start_write_group', (b'quack/', b'a token'), b'error', (b'UnsuspendableWriteGroup',))
        repo.lock_write()
        repo.start_write_group()
        self.assertEqual(client._calls[-2:], [('_ensure_real',), ('start_write_group',)])

    def test_commit_write_group(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.lock_write', (b'quack/', b''), b'success', (b'ok', b'a token'))
        client.add_expected_call(b'Repository.start_write_group', (b'quack/', b'a token'), b'success', (b'ok', [b'token1']))
        client.add_expected_call(b'Repository.commit_write_group', (b'quack/', b'a token', [b'token1']), b'success', (b'ok',))
        repo.lock_write()
        repo.start_write_group()
        repo.commit_write_group()

    def test_abort_write_group(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.lock_write', (b'quack/', b''), b'success', (b'ok', b'a token'))
        client.add_expected_call(b'Repository.start_write_group', (b'quack/', b'a token'), b'success', (b'ok', [b'token1']))
        client.add_expected_call(b'Repository.abort_write_group', (b'quack/', b'a token', [b'token1']), b'success', (b'ok',))
        repo.lock_write()
        repo.start_write_group()
        repo.abort_write_group(False)

    def test_suspend_write_group(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        self.assertEqual([], repo.suspend_write_group())

    def test_resume_write_group(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.lock_write', (b'quack/', b''), b'success', (b'ok', b'a token'))
        client.add_expected_call(b'Repository.check_write_group', (b'quack/', b'a token', [b'token1']), b'success', (b'ok',))
        repo.lock_write()
        repo.resume_write_group(['token1'])