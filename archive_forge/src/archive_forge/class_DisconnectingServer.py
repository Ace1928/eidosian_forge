import socketserver
from .. import errors, tests
from ..bzr.tests import test_read_bundle
from ..directory_service import directories
from ..mergeable import read_mergeable_from_url
from . import test_server
class DisconnectingServer(test_server.TestingTCPServerInAThread):

    def __init__(self):
        super().__init__(('127.0.0.1', 0), test_server.TestingTCPServer, DisconnectingHandler)

    def get_url(self):
        """Return the url of the server"""
        return 'bzr://%s:%d/' % self.server.server_address