import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
class SmartClientMediumTests(tests.TestCase):
    """Tests for SmartClientMedium.

    We should create a test scenario for this: we need a server module that
    construct the test-servers (like make_loopsocket_and_medium), and the list
    of SmartClientMedium classes to test.
    """

    def make_loopsocket_and_medium(self):
        """Create a loopback socket for testing, and a medium aimed at it."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        client_medium = medium.SmartTCPClientMedium('127.0.0.1', port, 'base')
        return (sock, client_medium)

    def receive_bytes_on_server(self, sock, bytes):
        """Accept a connection on sock and read 3 bytes.

        The bytes are appended to the list bytes.

        :return: a Thread which is running to do the accept and recv.
        """

        def _receive_bytes_on_server():
            connection, address = sock.accept()
            bytes.append(osutils.recv_all(connection, 3))
            connection.close()
        t = threading.Thread(target=_receive_bytes_on_server)
        t.start()
        return t

    def test_construct_smart_simple_pipes_client_medium(self):
        client_medium = medium.SmartSimplePipesClientMedium(None, None, None)

    def test_simple_pipes_client_request_type(self):
        client_medium = medium.SmartSimplePipesClientMedium(None, None, None)
        request = client_medium.get_request()
        self.assertIsInstance(request, medium.SmartClientStreamMediumRequest)

    def test_simple_pipes_client_get_concurrent_requests(self):
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(None, output, 'base')
        request = client_medium.get_request()
        request.finished_writing()
        request.finished_reading()
        request2 = client_medium.get_request()
        request2.finished_writing()
        request2.finished_reading()

    def test_simple_pipes_client__accept_bytes_writes_to_writable(self):
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(None, output, 'base')
        client_medium._accept_bytes(b'abc')
        self.assertEqual(b'abc', output.getvalue())

    def test_simple_pipes__accept_bytes_subprocess_closed(self):
        p = subprocess.Popen([sys.executable, '-c', 'import sys\nsys.stdout.write(sys.stdin.read(4))\nsys.stdout.close()\n'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0)
        client_medium = medium.SmartSimplePipesClientMedium(p.stdout, p.stdin, 'base')
        client_medium._accept_bytes(b'abc\n')
        self.assertEqual(b'abc', client_medium._read_bytes(3))
        p.wait()
        self.assertRaises(errors.ConnectionReset, client_medium._accept_bytes, b'more')

    def test_simple_pipes__accept_bytes_pipe_closed(self):
        child_read, client_write = create_file_pipes()
        client_medium = medium.SmartSimplePipesClientMedium(None, client_write, 'base')
        client_medium._accept_bytes(b'abc\n')
        self.assertEqual(b'abc\n', child_read.read(4))
        child_read.close()
        self.assertRaises(errors.ConnectionReset, client_medium._accept_bytes, b'more')

    def test_simple_pipes__flush_pipe_closed(self):
        child_read, client_write = create_file_pipes()
        client_medium = medium.SmartSimplePipesClientMedium(None, client_write, 'base')
        client_medium._accept_bytes(b'abc\n')
        child_read.close()
        client_medium._flush()

    def test_simple_pipes__flush_subprocess_closed(self):
        p = subprocess.Popen([sys.executable, '-c', 'import sys\nsys.stdout.write(sys.stdin.read(4))\nsys.stdout.close()\n'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0)
        client_medium = medium.SmartSimplePipesClientMedium(p.stdout, p.stdin, 'base')
        client_medium._accept_bytes(b'abc\n')
        p.wait()
        client_medium._flush()

    def test_simple_pipes__read_bytes_pipe_closed(self):
        child_read, client_write = create_file_pipes()
        client_medium = medium.SmartSimplePipesClientMedium(child_read, client_write, 'base')
        client_medium._accept_bytes(b'abc\n')
        client_write.close()
        self.assertEqual(b'abc\n', client_medium._read_bytes(4))
        self.assertEqual(b'', client_medium._read_bytes(4))

    def test_simple_pipes__read_bytes_subprocess_closed(self):
        p = subprocess.Popen([sys.executable, '-c', 'import sys\nif sys.platform == "win32":\n    import msvcrt, os\n    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)\n    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)\nsys.stdout.write(sys.stdin.read(4))\nsys.stdout.close()\n'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0)
        client_medium = medium.SmartSimplePipesClientMedium(p.stdout, p.stdin, 'base')
        client_medium._accept_bytes(b'abc\n')
        p.wait()
        self.assertEqual(b'abc\n', client_medium._read_bytes(4))
        self.assertEqual(b'', client_medium._read_bytes(4))

    def test_simple_pipes_client_disconnect_does_nothing(self):
        input = BytesIO()
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        client_medium._accept_bytes(b'abc')
        client_medium.disconnect()
        self.assertFalse(input.closed)
        self.assertFalse(output.closed)

    def test_simple_pipes_client_accept_bytes_after_disconnect(self):
        input = BytesIO()
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        client_medium._accept_bytes(b'abc')
        client_medium.disconnect()
        client_medium._accept_bytes(b'abc')
        self.assertFalse(input.closed)
        self.assertFalse(output.closed)
        self.assertEqual(b'abcabc', output.getvalue())

    def test_simple_pipes_client_ignores_disconnect_when_not_connected(self):
        client_medium = medium.SmartSimplePipesClientMedium(None, None, 'base')
        client_medium.disconnect()

    def test_simple_pipes_client_can_always_read(self):
        input = BytesIO(b'abcdef')
        client_medium = medium.SmartSimplePipesClientMedium(input, None, 'base')
        self.assertEqual(b'abc', client_medium.read_bytes(3))
        client_medium.disconnect()
        self.assertEqual(b'def', client_medium.read_bytes(3))

    def test_simple_pipes_client_supports__flush(self):
        from io import BytesIO
        input = BytesIO()
        output = BytesIO()
        flush_calls = []

        def logging_flush():
            flush_calls.append('flush')
        output.flush = logging_flush
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        client_medium._accept_bytes(b'abc')
        client_medium._flush()
        client_medium.disconnect()
        self.assertEqual(['flush'], flush_calls)

    def test_construct_smart_ssh_client_medium(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        unopened_port = sock.getsockname()[1]
        ssh_params = medium.SSHParams('127.0.0.1', unopened_port, None, None)
        client_medium = medium.SmartSSHClientMedium('base', ssh_params, 'not a vendor')
        sock.close()

    def test_ssh_client_connects_on_first_use(self):
        output = BytesIO()
        vendor = BytesIOSSHVendor(BytesIO(), output)
        ssh_params = medium.SSHParams('a hostname', 'a port', 'a username', 'a password', 'bzr')
        client_medium = medium.SmartSSHClientMedium('base', ssh_params, vendor)
        client_medium._accept_bytes(b'abc')
        self.assertEqual(b'abc', output.getvalue())
        self.assertEqual([('connect_ssh', 'a username', 'a password', 'a hostname', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes'])], vendor.calls)

    def test_ssh_client_changes_command_when_bzr_remote_path_passed(self):
        output = BytesIO()
        vendor = BytesIOSSHVendor(BytesIO(), output)
        ssh_params = medium.SSHParams('a hostname', 'a port', 'a username', 'a password', bzr_remote_path='fugly')
        client_medium = medium.SmartSSHClientMedium('base', ssh_params, vendor)
        client_medium._accept_bytes(b'abc')
        self.assertEqual(b'abc', output.getvalue())
        self.assertEqual([('connect_ssh', 'a username', 'a password', 'a hostname', 'a port', ['fugly', 'serve', '--inet', '--directory=/', '--allow-writes'])], vendor.calls)

    def test_ssh_client_disconnect_does_so(self):
        input = BytesIO()
        output = BytesIO()
        vendor = BytesIOSSHVendor(input, output)
        client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams('a hostname'), vendor)
        client_medium._accept_bytes(b'abc')
        client_medium.disconnect()
        self.assertTrue(input.closed)
        self.assertTrue(output.closed)
        self.assertEqual([('connect_ssh', None, None, 'a hostname', None, ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',)], vendor.calls)

    def test_ssh_client_disconnect_allows_reconnection(self):
        input = BytesIO()
        output = BytesIO()
        vendor = BytesIOSSHVendor(input, output)
        client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams('a hostname'), vendor)
        client_medium._accept_bytes(b'abc')
        client_medium.disconnect()
        input2 = BytesIO()
        output2 = BytesIO()
        vendor.read_from = input2
        vendor.write_to = output2
        client_medium._accept_bytes(b'abc')
        client_medium.disconnect()
        self.assertTrue(input.closed)
        self.assertTrue(output.closed)
        self.assertTrue(input2.closed)
        self.assertTrue(output2.closed)
        self.assertEqual([('connect_ssh', None, None, 'a hostname', None, ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',), ('connect_ssh', None, None, 'a hostname', None, ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',)], vendor.calls)

    def test_ssh_client_repr(self):
        client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams('example.com', '4242', 'username'))
        self.assertEqual('SmartSSHClientMedium(bzr+ssh://username@example.com:4242/)', repr(client_medium))

    def test_ssh_client_repr_no_port(self):
        client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams('example.com', None, 'username'))
        self.assertEqual('SmartSSHClientMedium(bzr+ssh://username@example.com/)', repr(client_medium))

    def test_ssh_client_repr_no_username(self):
        client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams('example.com', None, None))
        self.assertEqual('SmartSSHClientMedium(bzr+ssh://example.com/)', repr(client_medium))

    def test_ssh_client_ignores_disconnect_when_not_connected(self):
        client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams(None))
        client_medium.disconnect()

    def test_ssh_client_raises_on_read_when_not_connected(self):
        client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams(None))
        self.assertRaises(errors.MediumNotConnected, client_medium.read_bytes, 0)
        self.assertRaises(errors.MediumNotConnected, client_medium.read_bytes, 1)

    def test_ssh_client_supports__flush(self):
        from io import BytesIO
        input = BytesIO()
        output = BytesIO()
        flush_calls = []

        def logging_flush():
            flush_calls.append('flush')
        output.flush = logging_flush
        vendor = BytesIOSSHVendor(input, output)
        client_medium = medium.SmartSSHClientMedium('base', medium.SSHParams('a hostname'), vendor=vendor)
        client_medium._accept_bytes(b'abc')
        client_medium._flush()
        client_medium.disconnect()
        self.assertEqual(['flush'], flush_calls)

    def test_construct_smart_tcp_client_medium(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        unopened_port = sock.getsockname()[1]
        client_medium = medium.SmartTCPClientMedium('127.0.0.1', unopened_port, 'base')
        sock.close()

    def test_tcp_client_connects_on_first_use(self):
        sock, medium = self.make_loopsocket_and_medium()
        bytes = []
        t = self.receive_bytes_on_server(sock, bytes)
        medium.accept_bytes(b'abc')
        t.join()
        sock.close()
        self.assertEqual([b'abc'], bytes)

    def test_tcp_client_disconnect_does_so(self):
        sock, medium = self.make_loopsocket_and_medium()
        bytes = []
        t = self.receive_bytes_on_server(sock, bytes)
        medium.accept_bytes(b'ab')
        medium.disconnect()
        t.join()
        sock.close()
        self.assertEqual([b'ab'], bytes)
        medium.disconnect()

    def test_tcp_client_ignores_disconnect_when_not_connected(self):
        client_medium = medium.SmartTCPClientMedium(None, None, None)
        client_medium.disconnect()

    def test_tcp_client_raises_on_read_when_not_connected(self):
        client_medium = medium.SmartTCPClientMedium(None, None, None)
        self.assertRaises(errors.MediumNotConnected, client_medium.read_bytes, 0)
        self.assertRaises(errors.MediumNotConnected, client_medium.read_bytes, 1)

    def test_tcp_client_supports__flush(self):
        sock, medium = self.make_loopsocket_and_medium()
        bytes = []
        t = self.receive_bytes_on_server(sock, bytes)
        medium._flush()
        medium._accept_bytes(b'ab')
        medium._flush()
        medium.disconnect()
        t.join()
        sock.close()
        self.assertEqual([b'ab'], bytes)
        medium.disconnect()

    def test_tcp_client_host_unknown_connection_error(self):
        self.requireFeature(InvalidHostnameFeature)
        client_medium = medium.SmartTCPClientMedium('non_existent.invalid', 4155, 'base')
        self.assertRaises(errors.ConnectionError, client_medium._ensure_connection)