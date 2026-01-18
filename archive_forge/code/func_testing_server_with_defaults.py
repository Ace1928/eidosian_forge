import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
@pytest.fixture
def testing_server_with_defaults(wsgi_server_client):
    """Attach a WSGI app to the given server and preconfigure it."""
    wsgi_server = wsgi_server_client.server_instance
    wsgi_server.wsgi_app = HelloController()
    wsgi_server.server_client = wsgi_server_client
    return wsgi_server