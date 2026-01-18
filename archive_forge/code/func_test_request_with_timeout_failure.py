import time
import flask  # type: ignore
import pytest  # type: ignore
from pytest_localserver.http import WSGIServer  # type: ignore
from six.moves import http_client
from google.auth import exceptions
def test_request_with_timeout_failure(self, server):
    request = self.make_request()
    with pytest.raises(exceptions.TransportError):
        request(url=server.url + '/wait', method='GET', timeout=1)