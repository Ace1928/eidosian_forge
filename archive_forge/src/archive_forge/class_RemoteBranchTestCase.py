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
class RemoteBranchTestCase(RemoteBzrDirTestCase):

    def lock_remote_branch(self, branch):
        """Trick a RemoteBranch into thinking it is locked."""
        branch._lock_mode = 'w'
        branch._lock_count = 2
        branch._lock_token = b'branch token'
        branch._repo_lock_token = b'repo token'
        branch.repository._lock_mode = 'w'
        branch.repository._lock_count = 2
        branch.repository._lock_token = b'repo token'

    def make_remote_branch(self, transport, client):
        """Make a RemoteBranch using 'client' as its _SmartClient.

        A RemoteBzrDir and RemoteRepository will also be created to fill out
        the RemoteBranch, albeit with stub values for some of their attributes.
        """
        bzrdir = self.make_remote_bzrdir(transport, False)
        repo = RemoteRepository(bzrdir, None, _client=client)
        branch_format = self.get_branch_format()
        format = RemoteBranchFormat(network_name=branch_format.network_name())
        return RemoteBranch(bzrdir, repo, _client=client, format=format)