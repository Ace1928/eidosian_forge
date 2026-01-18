import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_parse_uri_unsafe_uri(test_client):
    """Test that malicious URI does not allow HTTP injection.

    This effectively checks that sending GET request with URL

    /%A0%D0blah%20key%200%20900%204%20data

    is not converted into

    GET /
    blah key 0 900 4 data
    HTTP/1.1

    which would be a security issue otherwise.
    """
    c = test_client.get_connection()
    resource = '/\xa0Ðblah key 0 900 4 data'.encode('latin-1')
    quoted = urllib.parse.quote(resource)
    assert quoted == '/%A0%D0blah%20key%200%20900%204%20data'
    request = 'GET {quoted} HTTP/1.1'.format(**locals())
    c._output(request.encode('utf-8'))
    c._send_output()
    response = _get_http_response(c, method='GET')
    response.begin()
    assert response.status == HTTP_OK
    assert response.read(12) == b'Hello world!'
    c.close()