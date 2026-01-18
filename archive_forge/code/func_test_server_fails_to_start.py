import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_server_fails_to_start(self):

    class CantStart(Exception):
        pass

    class CantStartServer(test_server.TestingTCPServer):

        def server_bind(self):
            raise CantStart()
    self.assertRaises(CantStart, self.get_server, server_class=CantStartServer)