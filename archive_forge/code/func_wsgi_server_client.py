import threading
import time
import pytest
from .._compat import IS_MACOS, IS_WINDOWS  # noqa: WPS436
from ..server import Gateway, HTTPServer
from ..testing import (  # noqa: F401  # pylint: disable=unused-import
from ..testing import get_server_client
@pytest.fixture
def wsgi_server_client(wsgi_server):
    """Create a test client out of given WSGI server."""
    return get_server_client(wsgi_server)