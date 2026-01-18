from contextlib import closing, contextmanager
import errno
import socket
import threading
import time
import http.client
import pytest
import cheroot.server
from cheroot.test import webtest
import cheroot.wsgi
@pytest.fixture
def wsgi_server():
    """Set up and tear down a Cheroot WSGI server instance."""
    with cheroot_server(cheroot.wsgi.Server) as srv:
        yield srv