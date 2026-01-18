import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_parse_uri_invalid_uri(test_client):
    """Check that server responds with Bad Request to invalid GET queries.

    Invalid request line test case: it should only contain US-ASCII.
    """
    c = test_client.get_connection()
    c._output(u'GET /йопта! HTTP/1.1'.encode('utf-8'))
    c._send_output()
    response = _get_http_response(c, method='GET')
    response.begin()
    assert response.status == HTTP_BAD_REQUEST
    assert response.read(21) == b'Malformed Request-URI'
    c.close()