import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_parse_uri_asterisk_uri(test_client):
    """Check that server responds with OK to OPTIONS with "*" Absolute URI."""
    status_line, _, actual_resp_body = test_client.options('*')
    actual_status = int(status_line[:3])
    assert actual_status == HTTP_OK
    expected_body = b'Got asterisk URI path with OPTIONS method'
    assert actual_resp_body == expected_body