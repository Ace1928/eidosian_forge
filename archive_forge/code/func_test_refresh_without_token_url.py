import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import exceptions
from google.auth import external_account_authorized_user
from google.auth import transport
def test_refresh_without_token_url(self):
    request = self.make_mock_request()
    creds = self.make_credentials(token_url=None, token=ACCESS_TOKEN)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        creds.refresh(request)
    assert excinfo.match('The credentials do not contain the necessary fields need to refresh the access token. You must specify refresh_token, token_url, client_id, and client_secret.')
    assert not creds.expiry
    assert not creds.expired
    assert not creds.requires_scopes
    assert creds.is_user
    request.assert_not_called()