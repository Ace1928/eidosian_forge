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
class TestBzrDirCreateRepository(TestRemote):

    def test_backwards_compat(self):
        self.setup_smart_server_with_call_log()
        bzrdir = self.make_controldir('.')
        self.reset_smart_call_log()
        self.disable_verb(b'BzrDir.create_repository')
        bzrdir.create_repository()
        create_repo_call_count = len([call for call in self.hpss_calls if call.call.method == b'BzrDir.create_repository'])
        self.assertEqual(1, create_repo_call_count)

    def test_current_server(self):
        transport = self.get_transport('.')
        transport = transport.clone('quack')
        self.make_controldir('quack')
        client = FakeClient(transport.base)
        reference_bzrdir_format = controldir.format_registry.get('default')()
        reference_format = reference_bzrdir_format.repository_format
        network_name = reference_format.network_name()
        client.add_expected_call(b'BzrDir.create_repository', (b'quack/', b'Bazaar repository format 2a (needs bzr 1.16 or later)\n', b'False'), b'success', (b'ok', b'yes', b'yes', b'yes', network_name))
        a_controldir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        repo = a_controldir.create_repository()
        self.assertIsInstance(repo, remote.RemoteRepository)
        format = repo._format
        self.assertTrue(format.rich_root_data)
        self.assertTrue(format.supports_tree_reference)
        self.assertTrue(format.supports_external_lookups)
        self.assertEqual(network_name, format.network_name())