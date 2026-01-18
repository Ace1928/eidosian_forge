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
def test_search_everything(self):
    """A search of 'everything' returns a stream."""
    backing = self.get_transport()
    request = smart_repo.SmartServerRepositoryGetStream_1_19(backing)
    repo, r1, r2 = self.make_two_commit_repo()
    serialised_fetch_spec = b'everything'
    request.execute(b'', repo._format.network_name())
    response = request.do_body(serialised_fetch_spec)
    self.assertEqual((b'ok',), response.args)
    stream_bytes = b''.join(response.body_stream)
    self.assertStartsWith(stream_bytes, b'Bazaar pack format 1')