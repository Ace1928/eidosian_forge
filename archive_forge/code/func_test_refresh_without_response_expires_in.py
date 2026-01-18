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
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min)
def test_refresh_without_response_expires_in(self, unused_utcnow):
    response = SUCCESS_RESPONSE.copy()
    del response['expires_in']
    expected_expires_in = 1800
    source_credentials = SourceCredentials(expires_in=expected_expires_in)
    expected_expiry = datetime.datetime.min + datetime.timedelta(seconds=expected_expires_in)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    request_data = {'grant_type': GRANT_TYPE, 'subject_token': 'ACCESS_TOKEN_1', 'subject_token_type': SUBJECT_TOKEN_TYPE, 'requested_token_type': REQUESTED_TOKEN_TYPE, 'options': urllib.parse.quote(json.dumps(CREDENTIAL_ACCESS_BOUNDARY_JSON))}
    request = self.make_mock_request(status=http_client.OK, data=response)
    credentials = self.make_credentials(source_credentials=source_credentials)
    with mock.patch.object(source_credentials, 'refresh', wraps=source_credentials.refresh) as wrapped_souce_cred_refresh:
        credentials.refresh(request)
        self.assert_request_kwargs(request.call_args[1], headers, request_data)
        assert credentials.valid
        assert credentials.expiry == expected_expiry
        assert not credentials.expired
        assert credentials.token == response['access_token']
        wrapped_souce_cred_refresh.assert_called_with(request)