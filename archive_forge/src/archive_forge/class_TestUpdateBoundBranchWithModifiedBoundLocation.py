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
class TestUpdateBoundBranchWithModifiedBoundLocation(tests.TestCaseWithTransport):
    """Ensure correct handling of bound_location modifications.

    This is tested against a smart server as http://pad.lv/786980 was about a
    ReadOnlyError (write attempt during a read-only transaction) which can only
    happen in this context.
    """

    def setUp(self):
        super().setUp()
        self.transport_server = test_server.SmartTCPServer_for_testing

    def make_master_and_checkout(self, master_name, checkout_name):
        self.master = self.make_branch_and_tree(master_name)
        self.checkout = self.master.branch.create_checkout(checkout_name)
        self.master.commit('add stuff')
        self.last_revid = self.master.commit('even more stuff')
        self.bound_location = self.checkout.branch.get_bound_location()

    def assertUpdateSucceeds(self, new_location):
        self.checkout.branch.set_bound_location(new_location)
        self.checkout.update()
        self.assertEqual(self.last_revid, self.checkout.last_revision())

    def test_without_final_slash(self):
        self.make_master_and_checkout('master', 'checkout')
        self.assertEndsWith(self.bound_location, '/')
        self.assertUpdateSucceeds(self.bound_location.rstrip('/'))

    def test_plus_sign(self):
        self.make_master_and_checkout('+master', 'checkout')
        self.assertUpdateSucceeds(self.bound_location.replace('%2B', '+', 1))

    def test_tilda(self):
        self.make_master_and_checkout('mas~ter', 'checkout')
        self.assertUpdateSucceeds(self.bound_location.replace('%2E', '~', 1))