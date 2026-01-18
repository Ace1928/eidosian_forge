import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import sts
from google.oauth2 import utils
def test_refresh_token_failure(self):
    """Test refresh token with failure response."""
    client = self.make_client(self.CLIENT_AUTH_BASIC)
    request = self.make_mock_request(status=http_client.BAD_REQUEST, data=self.ERROR_RESPONSE)
    with pytest.raises(exceptions.OAuthError) as excinfo:
        client.refresh_token(request, 'refreshtoken')
    assert excinfo.match('Error code invalid_request: Invalid subject token - https://tools.ietf.org/html/rfc6749')