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
class TestSmartServerRequestBzrDirInitializeEx(tests.TestCaseWithMemoryTransport):
    """Basic tests for BzrDir.initialize_ex_1.16 in the smart server.

    The main unit tests in test_bzrdir exercise the API comprehensively.
    """

    def test_empty_dir(self):
        """Initializing an empty dir should succeed and do it."""
        backing = self.get_transport()
        name = self.make_controldir('reference')._format.network_name()
        request = smart_dir.SmartServerRequestBzrDirInitializeEx(backing)
        self.assertEqual(smart_req.SmartServerResponse((b'', b'', b'', b'', b'', b'', name, b'False', b'', b'', b'')), request.execute(name, b'', b'True', b'False', b'False', b'', b'', b'', b'', b'False'))
        made_dir = controldir.ControlDir.open_from_transport(backing)
        self.assertRaises(errors.NoWorkingTree, made_dir.open_workingtree)
        self.assertRaises(errors.NotBranchError, made_dir.open_branch)
        self.assertRaises(errors.NoRepositoryPresent, made_dir.open_repository)

    def test_missing_dir(self):
        """Initializing a missing directory should fail like the bzrdir api."""
        backing = self.get_transport()
        name = self.make_controldir('reference')._format.network_name()
        request = smart_dir.SmartServerRequestBzrDirInitializeEx(backing)
        self.assertRaises(transport.NoSuchFile, request.execute, name, b'subdir/dir', b'False', b'False', b'False', b'', b'', b'', b'', b'False')

    def test_initialized_dir(self):
        """Initializing an extant directory should fail like the bzrdir api."""
        backing = self.get_transport()
        name = self.make_controldir('reference')._format.network_name()
        request = smart_dir.SmartServerRequestBzrDirInitializeEx(backing)
        self.make_controldir('subdir')
        self.assertRaises(transport.FileExists, request.execute, name, b'subdir', b'False', b'False', b'False', b'', b'', b'', b'', b'False')