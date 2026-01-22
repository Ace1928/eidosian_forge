import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
class CantStartServer(test_server.TestingTCPServer):

    def server_bind(self):
        raise CantStart()