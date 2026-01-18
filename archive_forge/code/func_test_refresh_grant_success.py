import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test_refresh_grant_success():
    with mock.patch('google.oauth2._client._token_endpoint_request_no_throw') as mock_token_request:
        mock_token_request.side_effect = [(False, {'error': 'invalid_grant', 'error_subtype': 'rapt_required'}, True), (True, {'access_token': 'access_token'}, None)]
        with mock.patch('google.oauth2.reauth.get_rapt_token', return_value='new_rapt_token'):
            assert reauth.refresh_grant(MOCK_REQUEST, 'token_uri', 'refresh_token', 'client_id', 'client_secret', enable_reauth_refresh=True) == ('access_token', 'refresh_token', None, {'access_token': 'access_token'}, 'new_rapt_token')