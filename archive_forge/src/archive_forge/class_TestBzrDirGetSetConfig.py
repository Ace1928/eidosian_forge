import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
class TestBzrDirGetSetConfig(RemoteBzrDirTestCase):

    def test__get_config(self):
        client = FakeClient()
        client.add_success_response_with_body(b'default_stack_on = /\n', b'ok')
        transport = MemoryTransport()
        bzrdir = self.make_remote_bzrdir(transport, client)
        config = bzrdir.get_config()
        self.assertEqual('/', config.get_default_stack_on())
        self.assertEqual([('call_expecting_body', b'BzrDir.get_config_file', (b'memory:///',))], client._calls)

    def test_set_option_uses_vfs(self):
        self.setup_smart_server_with_call_log()
        bzrdir = self.make_controldir('.')
        self.reset_smart_call_log()
        config = bzrdir.get_config()
        config.set_default_stack_on('/')
        self.assertLength(4, self.hpss_calls)

    def test_backwards_compat_get_option(self):
        self.setup_smart_server_with_call_log()
        bzrdir = self.make_controldir('.')
        verb = b'BzrDir.get_config_file'
        self.disable_verb(verb)
        self.reset_smart_call_log()
        self.assertEqual(None, bzrdir._get_config().get_option('default_stack_on'))
        self.assertLength(4, self.hpss_calls)