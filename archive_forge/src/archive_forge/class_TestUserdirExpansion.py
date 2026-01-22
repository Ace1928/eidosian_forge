import signal
import sys
import threading
from _thread import interrupt_main  # type: ignore
from ... import builtins, config, errors, osutils
from ... import revision as _mod_revision
from ... import trace, transport, urlutils
from ...branch import Branch
from ...bzr.smart import client, medium
from ...bzr.smart.server import BzrServerFactory, SmartTCPServer
from ...controldir import ControlDir
from ...transport import remote
from .. import TestCaseWithMemoryTransport, TestCaseWithTransport
class TestUserdirExpansion(TestCaseWithMemoryTransport):

    @staticmethod
    def fake_expanduser(path):
        """A simple, environment-independent, function for the duration of this
        test.

        Paths starting with a path segment of '~user' will expand to start with
        '/home/user/'.  Every other path will be unchanged.
        """
        if path.split('/', 1)[0] == '~user':
            return '/home/user' + path[len('~user'):]
        return path

    def make_test_server(self, base_path='/'):
        """Make and start a BzrServerFactory, backed by a memory transport, and
        creat '/home/user' in that transport.
        """
        bzr_server = BzrServerFactory(self.fake_expanduser, lambda t: base_path)
        mem_transport = self.get_transport()
        mem_transport.mkdir('home')
        mem_transport.mkdir('home/user')
        bzr_server.set_up(mem_transport, None, None, inet=True, timeout=4.0)
        self.addCleanup(bzr_server.tear_down)
        return bzr_server

    def test_bzr_serve_expands_userdir(self):
        bzr_server = self.make_test_server()
        self.assertTrue(bzr_server.smart_server.backing_transport.has('~user'))

    def test_bzr_serve_does_not_expand_userdir_outside_base(self):
        bzr_server = self.make_test_server('/foo')
        self.assertFalse(bzr_server.smart_server.backing_transport.has('~user'))

    def test_get_base_path(self):
        """cmd_serve will turn the --directory option into a LocalTransport
        (optionally decorated with 'readonly+').  BzrServerFactory can
        determine the original --directory from that transport.
        """
        base_dir = osutils.abspath('/a/b/c') + '/'
        base_url = urlutils.local_path_to_url(base_dir) + '/'

        def capture_transport(transport, host, port, inet, timeout):
            self.bzr_serve_transport = transport
        cmd = builtins.cmd_serve()
        cmd.run(directory=base_dir, protocol=capture_transport)
        server_maker = BzrServerFactory()
        self.assertEqual('readonly+%s' % base_url, self.bzr_serve_transport.base)
        self.assertEqual(base_dir, server_maker.get_base_path(self.bzr_serve_transport))
        cmd.run(directory=base_dir, protocol=capture_transport, allow_writes=True)
        server_maker = BzrServerFactory()
        self.assertEqual(base_url, self.bzr_serve_transport.base)
        self.assertEqual(base_dir, server_maker.get_base_path(self.bzr_serve_transport))
        cmd.run(directory=base_url, protocol=capture_transport)
        server_maker = BzrServerFactory()
        self.assertEqual('readonly+%s' % base_url, self.bzr_serve_transport.base)
        self.assertEqual(base_dir, server_maker.get_base_path(self.bzr_serve_transport))