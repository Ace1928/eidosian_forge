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
class TestBzrDirDestroyRepository(TestRemote):

    def test_destroy_repository(self):
        transport = self.get_transport('quack')
        client = FakeClient(transport.base)
        (client.add_expected_call(b'BzrDir.destroy_repository', (b'quack/',), b'success', (b'ok',)),)
        a_controldir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        a_controldir.destroy_repository()
        self.assertFinished(client)