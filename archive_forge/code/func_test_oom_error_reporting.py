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
def test_oom_error_reporting(self):
    """An out-of-memory condition on the server is reported clearly"""
    transport_path = 'quack'
    repo, client = self.setup_fake_client_and_repository(transport_path)
    client.add_expected_call(b'PackRepository.autopack', (b'quack/',), b'error', (b'MemoryError',))
    err = self.assertRaises(errors.BzrError, repo.autopack)
    self.assertContainsRe(str(err), '^remote server out of mem')