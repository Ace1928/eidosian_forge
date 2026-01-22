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
class TestBzrServe(TestBzrServeBase):

    def setUp(self):
        super().setUp()
        self.disable_missing_extensions_warning()

    def test_server_exception_with_hook(self):
        """Catch exception from the server in the server_exception hook.

        We use ``run_bzr_serve_then_func`` without a ``func`` so the server
        will receive a KeyboardInterrupt exception we want to catch.
        """

        def hook(exception):
            if exception[0] is KeyboardInterrupt:
                sys.stderr.write(b'catching KeyboardInterrupt\n')
                return True
            else:
                return False
        SmartTCPServer.hooks.install_named_hook('server_exception', hook, 'test_server_except_hook hook')
        args = ['--listen', 'localhost', '--port', '0', '--quiet']
        out, err = self.run_bzr_serve_then_func(args, retcode=0)
        self.assertEqual('catching KeyboardInterrupt\n', err)

    def test_server_exception_no_hook(self):
        """test exception without hook returns error"""
        args = []
        out, err = self.run_bzr_serve_then_func(args, retcode=3)

    def assertInetServerShutsdownCleanly(self, process):
        """Shutdown the server process looking for errors."""
        process.stdin.close()
        process.stdin = None
        result = self.finish_brz_subprocess(process)
        self.assertEqual(b'', result[0])
        self.assertEqual(b'', result[1])

    def assertServerFinishesCleanly(self, process):
        """Shutdown the brz serve instance process looking for errors."""
        result = self.finish_brz_subprocess(process, retcode=3, send_signal=signal.SIGINT)
        self.assertEqual(b'', result[0])
        self.assertEqual(b'brz: interrupted\n', result[1])

    def make_read_requests(self, branch):
        """Do some read only requests."""
        with branch.lock_read():
            branch.repository.all_revision_ids()
            self.assertEqual(_mod_revision.NULL_REVISION, branch.last_revision())

    def start_server_inet(self, extra_options=()):
        """Start a brz server subprocess using the --inet option.

        :param extra_options: extra options to give the server.
        :return: a tuple with the brz process handle for passing to
            finish_brz_subprocess, a client for the server, and a transport.
        """
        args = ['serve', '--inet']
        args.extend(extra_options)
        process = self.start_brz_subprocess(args)
        url = 'bzr://localhost/'
        self.permit_url(url)
        client_medium = medium.SmartSimplePipesClientMedium(process.stdout, process.stdin, url)
        transport = remote.RemoteTransport(url, medium=client_medium)
        return (process, transport)

    def start_server_port(self, extra_options=()):
        """Start a brz server subprocess.

        :param extra_options: extra options to give the server.
        :return: a tuple with the brz process handle for passing to
            finish_brz_subprocess, and the base url for the server.
        """
        args = ['serve', '--listen', 'localhost', '--port', '0']
        args.extend(extra_options)
        process = self.start_brz_subprocess(args, skip_if_plan_to_signal=True)
        port_line = process.stderr.readline()
        prefix = b'listening on port: '
        self.assertStartsWith(port_line, prefix)
        port = int(port_line[len(prefix):])
        url = 'bzr://localhost:%d/' % port
        self.permit_url(url)
        return (process, url)

    def test_bzr_serve_quiet(self):
        self.make_branch('.')
        args = ['--listen', 'localhost', '--port', '0', '--quiet']
        out, err = self.run_bzr_serve_then_func(args, retcode=3)
        self.assertEqual('', out)
        self.assertEqual('', err)

    def test_bzr_serve_inet_readonly(self):
        """brz server should provide a read only filesystem by default."""
        process, transport = self.start_server_inet()
        self.assertRaises(errors.TransportNotPossible, transport.mkdir, 'adir')
        self.assertInetServerShutsdownCleanly(process)

    def test_bzr_serve_inet_readwrite(self):
        self.make_branch('.')
        process, transport = self.start_server_inet(['--allow-writes'])
        branch = ControlDir.open_from_transport(transport).open_branch()
        self.make_read_requests(branch)
        transport.mkdir('adir')
        self.assertInetServerShutsdownCleanly(process)

    def test_bzr_serve_port_readonly(self):
        """brz server should provide a read only filesystem by default."""
        process, url = self.start_server_port()
        t = transport.get_transport_from_url(url)
        self.assertRaises(errors.TransportNotPossible, t.mkdir, 'adir')
        self.assertServerFinishesCleanly(process)

    def test_bzr_serve_port_readwrite(self):
        self.make_branch('.')
        process, url = self.start_server_port(['--allow-writes'])
        branch = Branch.open(url)
        self.make_read_requests(branch)
        self.assertServerFinishesCleanly(process)

    def test_bzr_serve_supports_protocol(self):
        self.make_branch('.')
        process, url = self.start_server_port(['--allow-writes', '--protocol=bzr'])
        branch = Branch.open(url)
        self.make_read_requests(branch)
        self.assertServerFinishesCleanly(process)

    def test_bzr_serve_dhpss(self):
        self.make_branch('.')
        log_fname = self.test_dir + '/server.log'
        self.overrideEnv('BRZ_LOG', log_fname)
        process, transport = self.start_server_inet(['-Dhpss'])
        branch = ControlDir.open_from_transport(transport).open_branch()
        self.make_read_requests(branch)
        self.assertInetServerShutsdownCleanly(process)
        f = open(log_fname, 'rb')
        content = f.read()
        f.close()
        self.assertContainsRe(content, b'hpss request: \\[[0-9-]+\\]')

    def test_bzr_serve_supports_configurable_timeout(self):
        gs = config.GlobalStack()
        gs.set('serve.client_timeout', 0.2)
        gs.store.save()
        process, url = self.start_server_port()
        self.build_tree_contents([('a_file', b'contents\n')])
        t = transport.get_transport_from_url(url)
        self.assertEqual(b'contents\n', t.get_bytes('a_file'))
        m = t.get_smart_medium()
        m.read_bytes(1)
        err = process.stderr.readline()
        self.assertEqual(b'Connection Timeout: disconnecting client after 0.2 seconds\n', err)
        self.assertServerFinishesCleanly(process)

    def test_bzr_serve_supports_client_timeout(self):
        process, url = self.start_server_port(['--client-timeout=0.1'])
        self.build_tree_contents([('a_file', b'contents\n')])
        t = transport.get_transport_from_url(url)
        self.assertEqual(b'contents\n', t.get_bytes('a_file'))
        m = t.get_smart_medium()
        m.read_bytes(1)
        err = process.stderr.readline()
        self.assertEqual(b'Connection Timeout: disconnecting client after 0.1 seconds\n', err)
        self.assertServerFinishesCleanly(process)

    def test_bzr_serve_graceful_shutdown(self):
        big_contents = b'a' * 64 * 1024
        self.build_tree_contents([('bigfile', big_contents)])
        process, url = self.start_server_port(['--client-timeout=1.0'])
        t = transport.get_transport_from_url(url)
        m = t.get_smart_medium()
        c = client._SmartClient(m)
        resp, response_handler = c.call_expecting_body(b'get', b'bigfile')
        self.assertEqual((b'ok',), resp)
        process.send_signal(signal.SIGHUP)
        self.assertEqual(b'Requested to stop gracefully\n', process.stderr.readline())
        self.assertIn(process.stderr.readline(), (b'', b'Waiting for 1 client(s) to finish\n'))
        body = response_handler.read_body_bytes()
        if body != big_contents:
            self.fail('Failed to properly read the contents of "bigfile"')
        self.assertEqual(b'', m.read_bytes(1))
        self.assertEqual(0, process.wait())