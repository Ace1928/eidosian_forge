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
class TestSmartServerBzrDirRequestDestroyBranch(tests.TestCaseWithMemoryTransport):
    """Tests for BzrDir.destroy_branch."""

    def test_destroy_branch_default(self):
        """The default branch can be removed."""
        backing = self.get_transport()
        self.make_branch('.')
        request_class = smart_dir.SmartServerBzrDirRequestDestroyBranch
        request = request_class(backing)
        expected = smart_req.SuccessfulSmartServerResponse((b'ok',))
        self.assertEqual(expected, request.execute(b'', None))

    def test_destroy_branch_named(self):
        """A named branch can be removed."""
        backing = self.get_transport()
        dir = self.make_repository('.', format='development-colo').controldir
        dir.create_branch(name='branchname')
        request_class = smart_dir.SmartServerBzrDirRequestDestroyBranch
        request = request_class(backing)
        expected = smart_req.SuccessfulSmartServerResponse((b'ok',))
        self.assertEqual(expected, request.execute(b'', b'branchname'))

    def test_destroy_branch_missing(self):
        """An error is raised if the branch didn't exist."""
        backing = self.get_transport()
        self.make_controldir('.', format='development-colo')
        request_class = smart_dir.SmartServerBzrDirRequestDestroyBranch
        request = request_class(backing)
        expected = smart_req.FailedSmartServerResponse((b'nobranch',), None)
        self.assertEqual(expected, request.execute(b'', b'branchname'))