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
class TestGetParentMapAllowsNew(tests.TestCaseWithTransport):

    def test_allows_new_revisions(self):
        """get_parent_map's results can be updated by commit."""
        smart_server = test_server.SmartTCPServer_for_testing()
        self.start_server(smart_server)
        self.make_branch('branch')
        branch = Branch.open(smart_server.get_url() + '/branch')
        tree = branch.create_checkout('tree', lightweight=True)
        tree.lock_write()
        self.addCleanup(tree.unlock)
        graph = tree.branch.repository.get_graph()
        self.assertEqual({}, graph.get_parent_map([b'rev1']))
        tree.commit('message', rev_id=b'rev1')
        graph = tree.branch.repository.get_graph()
        self.assertEqual({b'rev1': (b'null:',)}, graph.get_parent_map([b'rev1']))