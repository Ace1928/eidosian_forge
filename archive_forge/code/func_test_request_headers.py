import time
import flask  # type: ignore
import pytest  # type: ignore
from pytest_localserver.http import WSGIServer  # type: ignore
from six.moves import http_client
from google.auth import exceptions
def test_request_headers(self, server):
    request = self.make_request()
    response = request(url=server.url + '/basic', method='GET', headers={'x-test-header': 'hello world'})
    assert response.status == http_client.OK
    assert response.headers['x-test-header'] == 'hello world'
    assert response.data == b'Basic Content'