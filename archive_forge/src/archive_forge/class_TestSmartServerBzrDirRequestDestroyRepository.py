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
class TestSmartServerBzrDirRequestDestroyRepository(tests.TestCaseWithMemoryTransport):
    """Tests for BzrDir.destroy_repository."""

    def test_destroy_repository_default(self):
        """The repository can be removed."""
        backing = self.get_transport()
        self.make_repository('.')
        request_class = smart_dir.SmartServerBzrDirRequestDestroyRepository
        request = request_class(backing)
        expected = smart_req.SuccessfulSmartServerResponse((b'ok',))
        self.assertEqual(expected, request.execute(b''))

    def test_destroy_repository_missing(self):
        """An error is raised if the repository didn't exist."""
        backing = self.get_transport()
        self.make_controldir('.')
        request_class = smart_dir.SmartServerBzrDirRequestDestroyRepository
        request = request_class(backing)
        expected = smart_req.FailedSmartServerResponse((b'norepository',), None)
        self.assertEqual(expected, request.execute(b''))