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
class TestBranchGetSetConfig(RemoteBranchTestCase):

    def test_get_branch_conf(self):
        client = FakeClient()
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'memory:///',), b'error', (b'NotStacked',))
        client.add_success_response_with_body(b'# config file body', b'ok')
        transport = MemoryTransport()
        branch = self.make_remote_branch(transport, client)
        config = branch.get_config()
        config.has_explicit_nickname()
        self.assertEqual([('call', b'Branch.get_stacked_on_url', (b'memory:///',)), ('call_expecting_body', b'Branch.get_config_file', (b'memory:///',))], client._calls)

    def test_get_multi_line_branch_conf(self):
        client = FakeClient()
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'memory:///',), b'error', (b'NotStacked',))
        client.add_success_response_with_body(b'a = 1\nb = 2\nc = 3\n', b'ok')
        transport = MemoryTransport()
        branch = self.make_remote_branch(transport, client)
        config = branch.get_config()
        self.assertEqual('2', config.get_user_option('b'))

    def test_set_option(self):
        client = FakeClient()
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'memory:///',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.lock_write', (b'memory:///', b'', b''), b'success', (b'ok', b'branch token', b'repo token'))
        client.add_expected_call(b'Branch.set_config_option', (b'memory:///', b'branch token', b'repo token', b'foo', b'bar', b''), b'success', ())
        client.add_expected_call(b'Branch.unlock', (b'memory:///', b'branch token', b'repo token'), b'success', (b'ok',))
        transport = MemoryTransport()
        branch = self.make_remote_branch(transport, client)
        branch.lock_write()
        config = branch._get_config()
        config.set_option('foo', 'bar')
        branch.unlock()
        self.assertFinished(client)

    def test_set_option_with_dict(self):
        client = FakeClient()
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'memory:///',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.lock_write', (b'memory:///', b'', b''), b'success', (b'ok', b'branch token', b'repo token'))
        encoded_dict_value = b'd5:ascii1:a11:unicode \xe2\x8c\x9a3:\xe2\x80\xbde'
        client.add_expected_call(b'Branch.set_config_option_dict', (b'memory:///', b'branch token', b'repo token', encoded_dict_value, b'foo', b''), b'success', ())
        client.add_expected_call(b'Branch.unlock', (b'memory:///', b'branch token', b'repo token'), b'success', (b'ok',))
        transport = MemoryTransport()
        branch = self.make_remote_branch(transport, client)
        branch.lock_write()
        config = branch._get_config()
        config.set_option({'ascii': 'a', 'unicode ⌚': '‽'}, 'foo')
        branch.unlock()
        self.assertFinished(client)

    def test_set_option_with_bool(self):
        client = FakeClient()
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'memory:///',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.lock_write', (b'memory:///', b'', b''), b'success', (b'ok', b'branch token', b'repo token'))
        client.add_expected_call(b'Branch.set_config_option', (b'memory:///', b'branch token', b'repo token', b'True', b'foo', b''), b'success', ())
        client.add_expected_call(b'Branch.unlock', (b'memory:///', b'branch token', b'repo token'), b'success', (b'ok',))
        transport = MemoryTransport()
        branch = self.make_remote_branch(transport, client)
        branch.lock_write()
        config = branch._get_config()
        config.set_option(True, 'foo')
        branch.unlock()
        self.assertFinished(client)

    def test_backwards_compat_set_option(self):
        self.setup_smart_server_with_call_log()
        branch = self.make_branch('.')
        verb = b'Branch.set_config_option'
        self.disable_verb(verb)
        branch.lock_write()
        self.addCleanup(branch.unlock)
        self.reset_smart_call_log()
        branch._get_config().set_option('value', 'name')
        self.assertLength(11, self.hpss_calls)
        self.assertEqual('value', branch._get_config().get_option('name'))

    def test_backwards_compat_set_option_with_dict(self):
        self.setup_smart_server_with_call_log()
        branch = self.make_branch('.')
        verb = b'Branch.set_config_option_dict'
        self.disable_verb(verb)
        branch.lock_write()
        self.addCleanup(branch.unlock)
        self.reset_smart_call_log()
        config = branch._get_config()
        value_dict = {'ascii': 'a', 'unicode ⌚': '‽'}
        config.set_option(value_dict, 'name')
        self.assertLength(11, self.hpss_calls)
        self.assertEqual(value_dict, branch._get_config().get_option('name'))