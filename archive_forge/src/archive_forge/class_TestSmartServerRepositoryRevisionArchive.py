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
class TestSmartServerRepositoryRevisionArchive(tests.TestCaseWithTransport):

    def test_get(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryRevisionArchive(backing)
        t = self.make_branch_and_tree('.')
        self.addCleanup(t.lock_write().unlock)
        self.build_tree_contents([('file', b'somecontents')])
        t.add(['file'], ids=[b'thefileid'])
        t.commit(rev_id=b'somerev', message='add file')
        response = request.execute(b'', b'somerev', b'tar', b'foo.tar', b'foo')
        self.assertTrue(response.is_successful())
        self.assertEqual(response.args, (b'ok',))
        b = BytesIO(b''.join(response.body_stream))
        with tarfile.open(mode='r', fileobj=b) as tf:
            self.assertEqual(['foo/file'], tf.getnames())