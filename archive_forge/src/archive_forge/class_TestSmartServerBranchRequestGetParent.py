import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
class TestSmartServerBranchRequestGetParent(tests.TestCaseWithMemoryTransport):

    def test_get_parent_none(self):
        self.make_branch('base')
        request = smart_branch.SmartServerBranchGetParent(self.get_transport())
        response = request.execute(b'base')
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'',)), response)

    def test_get_parent_something(self):
        base_branch = self.make_branch('base')
        base_branch.set_parent(self.get_url('foo'))
        request = smart_branch.SmartServerBranchGetParent(self.get_transport())
        response = request.execute(b'base')
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'../foo',)), response)