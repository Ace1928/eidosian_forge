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
class TestSmartServerBzrDirRequestGetConfigFile(tests.TestCaseWithMemoryTransport):
    """Tests for BzrDir.get_config_file."""

    def test_present(self):
        backing = self.get_transport()
        dir = self.make_controldir('.')
        dir.get_config().set_default_stack_on('/')
        local_result = dir._get_config()._get_config_file().read()
        request_class = smart_dir.SmartServerBzrDirRequestConfigFile
        request = request_class(backing)
        expected = smart_req.SuccessfulSmartServerResponse((), local_result)
        self.assertEqual(expected, request.execute(b''))

    def test_missing(self):
        backing = self.get_transport()
        self.make_controldir('.')
        request_class = smart_dir.SmartServerBzrDirRequestConfigFile
        request = request_class(backing)
        expected = smart_req.SuccessfulSmartServerResponse((), b'')
        self.assertEqual(expected, request.execute(b''))