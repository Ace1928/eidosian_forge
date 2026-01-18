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
def test_revid_no_committers(self):
    body = b'firstrev: 123456.300 3600\nlatestrev: 654231.400 0\nrevisions: 2\nsize: 18\n'
    transport_path = 'quick'
    revid = 'È'.encode()
    repo, client = self.setup_fake_client_and_repository(transport_path)
    client.add_success_response_with_body(body, b'ok')
    result = repo.gather_stats(revid)
    self.assertEqual([('call_expecting_body', b'Repository.gather_stats', (b'quick/', revid, b'no'))], client._calls)
    self.assertEqual({'revisions': 2, 'size': 18, 'firstrev': (123456.3, 3600), 'latestrev': (654231.4, 0)}, result)