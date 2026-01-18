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
def test_stacked_on_stacked_get_stream_unordered(self):

    def make_stacked_stacked():
        _, stacked = self.prepare_stacked_remote_branch()
        tree = stacked.controldir.sprout('tree3', stacked=True).open_workingtree()
        local_tree = tree.branch.create_checkout('local-tree3')
        local_tree.commit('more local changes are better')
        branch = Branch.open(self.get_url('tree3'))
        branch.lock_read()
        self.addCleanup(branch.unlock)
        return (None, branch)
    rev_ord, expected_revs = self.get_ordered_revs('1.9', 'unordered', branch_factory=make_stacked_stacked)
    self.assertEqual(set(expected_revs), set(rev_ord))
    self.assertLength(3, self.hpss_calls)