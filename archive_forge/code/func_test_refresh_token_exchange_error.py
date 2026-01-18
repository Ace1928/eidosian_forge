import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials
from google.auth import downscoped
from google.auth import exceptions
from google.auth import transport
def test_refresh_token_exchange_error(self):
    request = self.make_mock_request(status=http_client.BAD_REQUEST, data=ERROR_RESPONSE)
    credentials = self.make_credentials()
    with pytest.raises(exceptions.OAuthError) as excinfo:
        credentials.refresh(request)
    assert excinfo.match('Error code invalid_grant: Subject token is invalid. - https://tools.ietf.org/html/rfc6749')
    assert not credentials.expired
    assert credentials.token is None