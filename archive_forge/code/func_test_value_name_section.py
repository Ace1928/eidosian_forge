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
def test_value_name_section(self):
    branch = self.make_branch('.')
    request = smart_branch.SmartServerBranchRequestSetConfigOptionDict(branch.controldir.root_transport)
    branch_token, repo_token = self.get_lock_tokens(branch)
    config = branch._get_config()
    result = request.execute(b'', branch_token, repo_token, self.encoded_value_dict, b'foo', b'gam')
    self.assertEqual(smart_req.SuccessfulSmartServerResponse(()), result)
    self.assertEqual(self.value_dict, config.get_option('foo', 'gam'))
    branch.unlock()