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
class TestTransportMkdir(tests.TestCase):

    def test_permissiondenied(self):
        client = FakeClient()
        client.add_error_response(b'PermissionDenied', b'remote path', b'extra')
        transport = RemoteTransport('bzr://example.com/', medium=False, _client=client)
        exc = self.assertRaises(errors.PermissionDenied, transport.mkdir, 'client path')
        expected_error = errors.PermissionDenied('/client path', 'extra')
        self.assertEqual(expected_error, exc)