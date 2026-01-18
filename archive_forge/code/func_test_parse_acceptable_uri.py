import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
@pytest.mark.parametrize('uri', ('/hello', '/query_string?test=True', '/{0}?{1}={2}'.format(*map(urllib.parse.quote, ('Юххууу', 'ї', 'йо')))))
def test_parse_acceptable_uri(test_client, uri):
    """Check that server responds with OK to valid GET queries."""
    status_line = test_client.get(uri)[0]
    actual_status = int(status_line[:3])
    assert actual_status == HTTP_OK