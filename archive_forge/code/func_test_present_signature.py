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
def test_present_signature(self):
    """For a present signature, ('yes', ) is returned."""
    backing = self.get_transport()
    request = smart_repo.SmartServerRequestHasSignatureForRevisionId(backing)
    strategy = gpg.LoopbackGPGStrategy(None)
    tree = self.make_branch_and_memory_tree('.')
    tree.lock_write()
    tree.add('')
    tree.commit('a commit', rev_id=b'A')
    tree.branch.repository.start_write_group()
    tree.branch.repository.sign_revision(b'A', strategy)
    tree.branch.repository.commit_write_group()
    tree.unlock()
    self.assertTrue(tree.branch.repository.has_revision(b'A'))
    self.assertEqual(smart_req.SmartServerResponse((b'yes',)), request.execute(b'', b'A'))