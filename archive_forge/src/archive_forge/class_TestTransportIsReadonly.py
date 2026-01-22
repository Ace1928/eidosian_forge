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
class TestTransportIsReadonly(tests.TestCase):

    def test_true(self):
        client = FakeClient()
        client.add_success_response(b'yes')
        transport = RemoteTransport('bzr://example.com/', medium=False, _client=client)
        self.assertEqual(True, transport.is_readonly())
        self.assertEqual([('call', b'Transport.is_readonly', ())], client._calls)

    def test_false(self):
        client = FakeClient()
        client.add_success_response(b'no')
        transport = RemoteTransport('bzr://example.com/', medium=False, _client=client)
        self.assertEqual(False, transport.is_readonly())
        self.assertEqual([('call', b'Transport.is_readonly', ())], client._calls)

    def test_error_from_old_server(self):
        """bzr 0.15 and earlier servers don't recognise the is_readonly verb.

        Clients should treat it as a "no" response, because is_readonly is only
        advisory anyway (a transport could be read-write, but then the
        underlying filesystem could be readonly anyway).
        """
        client = FakeClient()
        client.add_unknown_method_response(b'Transport.is_readonly')
        transport = RemoteTransport('bzr://example.com/', medium=False, _client=client)
        self.assertEqual(False, transport.is_readonly())
        self.assertEqual([('call', b'Transport.is_readonly', ())], client._calls)