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
class TestBzrDirGetBranches(TestRemote):

    def test_get_branches(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        reference_bzrdir_format = controldir.format_registry.get('default')()
        branch_name = reference_bzrdir_format.get_branch_format().network_name()
        client.add_success_response_with_body(bencode.bencode({b'foo': (b'branch', branch_name), b'': (b'branch', branch_name)}), b'success')
        client.add_success_response(b'ok', b'', b'no', b'no', b'no', reference_bzrdir_format.repository_format.network_name())
        client.add_error_response(b'NotStacked')
        client.add_success_response(b'ok', b'', b'no', b'no', b'no', reference_bzrdir_format.repository_format.network_name())
        client.add_error_response(b'NotStacked')
        transport.mkdir('quack')
        transport = transport.clone('quack')
        a_controldir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        result = a_controldir.get_branches()
        self.assertEqual({'', 'foo'}, set(result.keys()))
        self.assertEqual([('call_expecting_body', b'BzrDir.get_branches', (b'quack/',)), ('call', b'BzrDir.find_repositoryV3', (b'quack/',)), ('call', b'Branch.get_stacked_on_url', (b'quack/',)), ('call', b'BzrDir.find_repositoryV3', (b'quack/',)), ('call', b'Branch.get_stacked_on_url', (b'quack/',))], client._calls)