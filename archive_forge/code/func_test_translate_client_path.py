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
def test_translate_client_path(self):
    transport = self.get_transport()
    request = smart_req.SmartServerRequest(transport, 'foo/')
    self.assertEqual('./', request.translate_client_path(b'foo/'))
    self.assertRaises(urlutils.InvalidURLJoin, request.translate_client_path, b'foo/..')
    self.assertRaises(errors.PathNotChild, request.translate_client_path, b'/')
    self.assertRaises(errors.PathNotChild, request.translate_client_path, b'bar/')
    self.assertEqual('./baz', request.translate_client_path(b'foo/baz'))
    e_acute = 'é'
    self.assertEqual('./' + urlutils.escape(e_acute), request.translate_client_path(b'foo/' + e_acute.encode('utf-8')))