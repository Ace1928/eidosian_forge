import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_parse_uri_fragment_uri(test_client):
    """Check that server responds with Bad Request to URI with fragment."""
    status_line, _, actual_resp_body = test_client.get('/hello?test=something#fake')
    actual_status = int(status_line[:3])
    assert actual_status == HTTP_BAD_REQUEST
    expected_body = b'Illegal #fragment in Request-URI.'
    assert actual_resp_body == expected_body