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
def test_refresh_workforce_impersonation_without_client_auth_success(self):
    expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=2800)).isoformat('T') + 'Z'
    expected_expiry = datetime.datetime.strptime(expire_time, '%Y-%m-%dT%H:%M:%SZ')
    token_response = self.SUCCESS_RESPONSE.copy()
    token_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    token_request_data = {'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange', 'audience': self.WORKFORCE_AUDIENCE, 'requested_token_type': 'urn:ietf:params:oauth:token-type:access_token', 'subject_token': 'subject_token_0', 'subject_token_type': self.WORKFORCE_SUBJECT_TOKEN_TYPE, 'scope': 'https://www.googleapis.com/auth/iam', 'options': urllib.parse.quote(json.dumps({'userProject': self.WORKFORCE_POOL_USER_PROJECT}))}
    impersonation_response = {'accessToken': 'SA_ACCESS_TOKEN', 'expireTime': expire_time}
    impersonation_headers = {'Content-Type': 'application/json', 'authorization': 'Bearer {}'.format(token_response['access_token'])}
    impersonation_request_data = {'delegates': None, 'scope': self.SCOPES, 'lifetime': '3600s'}
    request = self.make_mock_request(status=http_client.OK, data=token_response, impersonation_status=http_client.OK, impersonation_data=impersonation_response)
    credentials = self.make_workforce_pool_credentials(service_account_impersonation_url=self.SERVICE_ACCOUNT_IMPERSONATION_URL, scopes=self.SCOPES, workforce_pool_user_project=self.WORKFORCE_POOL_USER_PROJECT)
    credentials.refresh(request)
    assert len(request.call_args_list) == 2
    self.assert_token_request_kwargs(request.call_args_list[0][1], token_headers, token_request_data)
    self.assert_impersonation_request_kwargs(request.call_args_list[1][1], impersonation_headers, impersonation_request_data)
    assert credentials.valid
    assert credentials.expiry == expected_expiry
    assert not credentials.expired
    assert credentials.token == impersonation_response['accessToken']