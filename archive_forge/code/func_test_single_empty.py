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
def test_single_empty(self):
    transport_path = 'quack'
    repo, client = self.setup_fake_client_and_repository(transport_path)
    fmt = controldir.format_registry.get('2a')().repository_format
    repo._format = fmt
    stream = [('inventory-deltas', [versionedfile.FulltextContentFactory(b'somerevid', None, None, self._serialize_inv_delta(b'null:', b'somerevid', []))])]
    client.add_expected_call(b'VersionedFileRepository.get_inventories', (b'quack/', b'unordered'), b'success', (b'ok',), _stream_to_byte_stream(stream, fmt))
    ret = list(repo.iter_inventories([b'somerevid']))
    self.assertLength(1, ret)
    inv = ret[0]
    self.assertEqual(b'somerevid', inv.revision_id)