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
class TestRepositoryGetRevisionSignatureText(TestRemoteRepository):

    def test_text(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response_with_body(b'THETEXT', b'ok')
        self.assertEqual(b'THETEXT', repo.get_signature_text(b'revid'))
        self.assertEqual([('call_expecting_body', b'Repository.get_revision_signature_text', (b'quack/', b'revid'))], client._calls)

    def test_no_signature(self):
        transport_path = 'quick'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_error_response(b'nosuchrevision', b'unknown')
        self.assertRaises(errors.NoSuchRevision, repo.get_signature_text, b'unknown')
        self.assertEqual([('call_expecting_body', b'Repository.get_revision_signature_text', (b'quick/', b'unknown'))], client._calls)