import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_start_stop(self):
    server = self.get_server()
    client = self.get_client()
    server.stop_server()
    client = self.get_client()
    self.assertRaises(socket.error, client.connect, (server.host, server.port))