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
class SmartTCPServerHookTests(tests.TestCaseWithMemoryTransport):
    """Tests for SmartTCPServer hooks."""

    def setUp(self):
        super().setUp()
        self.server = server.SmartTCPServer(self.get_transport())

    def test_run_server_started_hooks(self):
        """Test the server started hooks get fired properly."""
        started_calls = []
        server.SmartTCPServer.hooks.install_named_hook('server_started', lambda backing_urls, url: started_calls.append((backing_urls, url)), None)
        started_ex_calls = []
        server.SmartTCPServer.hooks.install_named_hook('server_started_ex', lambda backing_urls, url: started_ex_calls.append((backing_urls, url)), None)
        self.server._sockname = ('example.com', 42)
        self.server.run_server_started_hooks()
        self.assertEqual(started_calls, [([self.get_transport().base], 'bzr://example.com:42/')])
        self.assertEqual(started_ex_calls, [([self.get_transport().base], self.server)])

    def test_run_server_started_hooks_ipv6(self):
        """Test that socknames can contain 4-tuples."""
        self.server._sockname = ('::', 42, 0, 0)
        started_calls = []
        server.SmartTCPServer.hooks.install_named_hook('server_started', lambda backing_urls, url: started_calls.append((backing_urls, url)), None)
        self.server.run_server_started_hooks()
        self.assertEqual(started_calls, [([self.get_transport().base], 'bzr://:::42/')])

    def test_run_server_stopped_hooks(self):
        """Test the server stopped hooks."""
        self.server._sockname = ('example.com', 42)
        stopped_calls = []
        server.SmartTCPServer.hooks.install_named_hook('server_stopped', lambda backing_urls, url: stopped_calls.append((backing_urls, url)), None)
        self.server.run_server_stopped_hooks()
        self.assertEqual(stopped_calls, [([self.get_transport().base], 'bzr://example.com:42/')])