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
def test_revid_with_committers(self):
    """For a revid we get more infos."""
    backing = self.get_transport()
    rev_id_utf8 = 'Èabc'.encode()
    request = smart_repo.SmartServerRepositoryGatherStats(backing)
    tree = self.make_branch_and_memory_tree('.')
    tree.lock_write()
    tree.add('')
    tree.commit('a commit', timestamp=123456.2, timezone=3600)
    tree.commit('a commit', timestamp=654321.4, timezone=0, rev_id=rev_id_utf8)
    tree.unlock()
    tree.branch.repository.gather_stats()
    expected_body = b'firstrev: 123456.200 3600\nlatestrev: 654321.400 0\nrevisions: 2\n'
    self.assertEqual(smart_req.SmartServerResponse((b'ok',), expected_body), request.execute(b'', rev_id_utf8, b'no'))