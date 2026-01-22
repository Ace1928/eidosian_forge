import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestTestingSmartConnectionHandler(tests.TestCase):

    def test_connection_timeout_suppressed(self):
        self.overrideAttr(test_server, '_DEFAULT_TESTING_CLIENT_TIMEOUT', 0.01)
        s = FakeServer()
        server_sock, client_sock = portable_socket_pair()
        test_server.TestingSmartConnectionHandler(server_sock, server_sock.getpeername(), s)

    def test_connection_shutdown_while_serving_no_error(self):
        s = FakeServer()
        server_sock, client_sock = portable_socket_pair()

        class ShutdownConnectionHandler(test_server.TestingSmartConnectionHandler):

            def _build_protocol(self):
                self.finished = True
                return super()._build_protocol()
        ShutdownConnectionHandler(server_sock, server_sock.getpeername(), s)