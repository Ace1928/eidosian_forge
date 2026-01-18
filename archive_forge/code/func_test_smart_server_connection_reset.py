import socketserver
from .. import errors, tests
from ..bzr.tests import test_read_bundle
from ..directory_service import directories
from ..mergeable import read_mergeable_from_url
from . import test_server
def test_smart_server_connection_reset(self):
    """If a smart server connection fails during the attempt to read a
        bundle, then the ConnectionReset error should be propagated.
        """
    sock_server = DisconnectingServer()
    self.start_server(sock_server)
    url = sock_server.get_url()
    self.assertRaises(errors.ConnectionReset, read_mergeable_from_url, url)