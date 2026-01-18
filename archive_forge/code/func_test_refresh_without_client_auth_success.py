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
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min)
def test_refresh_without_client_auth_success(self, unused_utcnow):
    response = self.SUCCESS_RESPONSE.copy()
    response['expires_in'] = 2800
    expected_expiry = datetime.datetime.min + datetime.timedelta(seconds=response['expires_in'])
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    request_data = {'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange', 'audience': self.AUDIENCE, 'requested_token_type': 'urn:ietf:params:oauth:token-type:access_token', 'subject_token': 'subject_token_0', 'subject_token_type': self.SUBJECT_TOKEN_TYPE}
    request = self.make_mock_request(status=http_client.OK, data=response)
    credentials = self.make_credentials()
    credentials.refresh(request)
    self.assert_token_request_kwargs(request.call_args[1], headers, request_data)
    assert credentials.valid
    assert credentials.expiry == expected_expiry
    assert not credentials.expired
    assert credentials.token == response['access_token']