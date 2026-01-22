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
class TestSmartServerRequestOpenBranchV2(TestCaseWithChrootedTransport):

    def test_no_branch(self):
        """When there is no branch, ('nobranch', ) is returned."""
        backing = self.get_transport()
        self.make_controldir('.')
        request = smart_dir.SmartServerRequestOpenBranchV2(backing)
        self.assertEqual(smart_req.SmartServerResponse((b'nobranch',)), request.execute(b''))

    def test_branch(self):
        """When there is a branch, 'ok' is returned."""
        backing = self.get_transport()
        expected = self.make_branch('.')._format.network_name()
        request = smart_dir.SmartServerRequestOpenBranchV2(backing)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'branch', expected)), request.execute(b''))

    def test_branch_reference(self):
        """When there is a branch reference, the reference URL is returned."""
        self.vfs_transport_factory = test_server.LocalURLServer
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestOpenBranchV2(backing)
        branch = self.make_branch('branch')
        checkout = branch.create_checkout('reference', lightweight=True)
        reference_url = _mod_bzrbranch.BranchReferenceFormat().get_reference(checkout.controldir).encode('utf-8')
        self.assertFileEqual(reference_url, 'reference/.bzr/branch/location')
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ref', reference_url)), request.execute(b'reference'))

    def test_stacked_branch(self):
        """Opening a stacked branch does not open the stacked-on branch."""
        trunk = self.make_branch('trunk')
        feature = self.make_branch('feature')
        feature.set_stacked_on_url(trunk.base)
        opened_branches = []
        _mod_branch.Branch.hooks.install_named_hook('open', opened_branches.append, None)
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestOpenBranchV2(backing)
        request.setup_jail()
        try:
            response = request.execute(b'feature')
        finally:
            request.teardown_jail()
        expected_format = feature._format.network_name()
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'branch', expected_format)), response)
        self.assertLength(1, opened_branches)

    def test_notification_on_branch_from_repository(self):
        """When there is a repository, the error should return details."""
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestOpenBranchV2(backing)
        self.make_repository('.')
        self.assertEqual(smart_req.SmartServerResponse((b'nobranch',)), request.execute(b''))