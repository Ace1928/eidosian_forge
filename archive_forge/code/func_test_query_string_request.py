import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_query_string_request(test_client):
    """Check that GET param is parsed well."""
    status_line, _, actual_resp_body = test_client.get('/query_string?test=True')
    actual_status = int(status_line[:3])
    assert actual_status == HTTP_OK
    assert actual_resp_body == b'test=True'