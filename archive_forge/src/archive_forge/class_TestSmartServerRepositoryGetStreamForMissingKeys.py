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
class TestSmartServerRepositoryGetStreamForMissingKeys(GetStreamTestBase):

    def test_missing(self):
        """The search argument may be a 'ancestry-of' some heads'."""
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetStreamForMissingKeys(backing)
        repo, r1, r2 = self.make_two_commit_repo()
        request.execute(b'', repo._format.network_name())
        lines = b'inventories\t' + r1
        response = request.do_body(lines)
        self.assertEqual((b'ok',), response.args)
        stream_bytes = b''.join(response.body_stream)
        self.assertStartsWith(stream_bytes, b'Bazaar pack format 1')

    def test_unknown_format(self):
        """The format may not be known by the remote server."""
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetStreamForMissingKeys(backing)
        repo, r1, r2 = self.make_two_commit_repo()
        request.execute(b'', b'yada yada yada')
        expected = smart_req.FailedSmartServerResponse((b'UnknownFormat', b'repository', b'yada yada yada'))