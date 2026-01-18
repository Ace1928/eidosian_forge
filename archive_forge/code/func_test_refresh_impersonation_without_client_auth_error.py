import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
from google.auth import transport
def test_refresh_impersonation_without_client_auth_error(self):
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE, impersonation_status=http_client.BAD_REQUEST, impersonation_data=self.IMPERSONATION_ERROR_RESPONSE)
    credentials = self.make_credentials(service_account_impersonation_url=self.SERVICE_ACCOUNT_IMPERSONATION_URL, scopes=self.SCOPES)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.refresh(request)
    assert excinfo.match('Unable to acquire impersonated credentials')
    assert not credentials.expired
    assert credentials.token is None