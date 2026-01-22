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
class TestBzrDirFormatInitializeEx(TestRemote):

    def test_success(self):
        """Simple test for typical successful call."""
        fmt = RemoteBzrDirFormat()
        default_format_name = BzrDirFormat.get_default_format().network_name()
        transport = self.get_transport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'BzrDirFormat.initialize_ex_1.16', (default_format_name, b'path', b'False', b'False', b'False', b'', b'', b'', b'', b'False'), b'success', (b'.', b'no', b'no', b'yes', b'repo fmt', b'repo bzrdir fmt', b'bzrdir fmt', b'False', b'', b'', b'repo lock token'))
        fmt._initialize_on_transport_ex_rpc(client, b'path', transport, False, False, False, None, None, None, None, False)
        self.assertFinished(client)

    def test_error(self):
        """Error responses are translated, e.g. 'PermissionDenied' raises the
        corresponding error from the client.
        """
        fmt = RemoteBzrDirFormat()
        default_format_name = BzrDirFormat.get_default_format().network_name()
        transport = self.get_transport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'BzrDirFormat.initialize_ex_1.16', (default_format_name, b'path', b'False', b'False', b'False', b'', b'', b'', b'', b'False'), b'error', (b'PermissionDenied', b'path', b'extra info'))
        err = self.assertRaises(errors.PermissionDenied, fmt._initialize_on_transport_ex_rpc, client, b'path', transport, False, False, False, None, None, None, None, False)
        self.assertEqual('path', err.path)
        self.assertEqual(': extra info', err.extra)
        self.assertFinished(client)

    def test_error_from_real_server(self):
        """Integration test for error translation."""
        transport = self.make_smart_server('foo')
        transport = transport.clone('no-such-path')
        fmt = RemoteBzrDirFormat()
        self.assertRaises(_mod_transport.NoSuchFile, fmt.initialize_on_transport_ex, transport, create_prefix=False)