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
class TestSmartServerBzrDirRequestCloningMetaDir(tests.TestCaseWithMemoryTransport):
    """Tests for BzrDir.cloning_metadir."""

    def test_cloning_metadir(self):
        """When there is a bzrdir present, the call succeeds."""
        backing = self.get_transport()
        dir = self.make_controldir('.')
        local_result = dir.cloning_metadir()
        request_class = smart_dir.SmartServerBzrDirRequestCloningMetaDir
        request = request_class(backing)
        expected = smart_req.SuccessfulSmartServerResponse((local_result.network_name(), local_result.repository_format.network_name(), (b'branch', local_result.get_branch_format().network_name())))
        self.assertEqual(expected, request.execute(b'', b'False'))

    def test_cloning_metadir_reference(self):
        """The request fails when bzrdir contains a branch reference."""
        backing = self.get_transport()
        referenced_branch = self.make_branch('referenced')
        dir = self.make_controldir('.')
        dir.cloning_metadir()
        _mod_bzrbranch.BranchReferenceFormat().initialize(dir, target_branch=referenced_branch)
        _mod_bzrbranch.BranchReferenceFormat().get_reference(dir)
        backing.rename('referenced', 'moved')
        request_class = smart_dir.SmartServerBzrDirRequestCloningMetaDir
        request = request_class(backing)
        expected = smart_req.FailedSmartServerResponse((b'BranchReference',))
        self.assertEqual(expected, request.execute(b'', b'False'))