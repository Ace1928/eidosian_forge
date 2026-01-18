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
def test_refresh_without_client_auth_success_explicit_default_scopes_only(self):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    request_data = {'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange', 'audience': self.AUDIENCE, 'requested_token_type': 'urn:ietf:params:oauth:token-type:access_token', 'scope': 'scope1 scope2', 'subject_token': 'subject_token_0', 'subject_token_type': self.SUBJECT_TOKEN_TYPE}
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE)
    credentials = self.make_credentials(scopes=None, default_scopes=['scope1', 'scope2'])
    credentials.refresh(request)
    self.assert_token_request_kwargs(request.call_args[1], headers, request_data)
    assert credentials.valid
    assert not credentials.expired
    assert credentials.token == self.SUCCESS_RESPONSE['access_token']
    assert credentials.has_scopes(['scope1', 'scope2'])