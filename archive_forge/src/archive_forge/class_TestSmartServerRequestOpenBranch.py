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
class TestSmartServerRequestOpenBranch(TestCaseWithChrootedTransport):

    def test_no_branch(self):
        """When there is no branch, ('nobranch', ) is returned."""
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestOpenBranch(backing)
        self.make_controldir('.')
        self.assertEqual(smart_req.SmartServerResponse((b'nobranch',)), request.execute(b''))

    def test_branch(self):
        """When there is a branch, 'ok' is returned."""
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestOpenBranch(backing)
        self.make_branch('.')
        self.assertEqual(smart_req.SmartServerResponse((b'ok', b'')), request.execute(b''))

    def test_branch_reference(self):
        """When there is a branch reference, the reference URL is returned."""
        self.vfs_transport_factory = test_server.LocalURLServer
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestOpenBranch(backing)
        branch = self.make_branch('branch')
        checkout = branch.create_checkout('reference', lightweight=True)
        reference_url = _mod_bzrbranch.BranchReferenceFormat().get_reference(checkout.controldir).encode('utf-8')
        self.assertFileEqual(reference_url, 'reference/.bzr/branch/location')
        self.assertEqual(smart_req.SmartServerResponse((b'ok', reference_url)), request.execute(b'reference'))

    def test_notification_on_branch_from_repository(self):
        """When there is a repository, the error should return details."""
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestOpenBranch(backing)
        self.make_repository('.')
        self.assertEqual(smart_req.SmartServerResponse((b'nobranch',)), request.execute(b''))