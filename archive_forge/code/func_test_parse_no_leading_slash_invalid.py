import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
@pytest.mark.parametrize('uri', ('hello', 'привіт'))
def test_parse_no_leading_slash_invalid(test_client, uri):
    """Check that server responds with Bad Request to invalid GET queries.

    Invalid request line test case: it should have leading slash (be absolute).
    """
    status_line, _, actual_resp_body = test_client.get(urllib.parse.quote(uri))
    actual_status = int(status_line[:3])
    assert actual_status == HTTP_BAD_REQUEST
    assert b'starting with a slash' in actual_resp_body