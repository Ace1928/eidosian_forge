import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import id_token
from google.oauth2 import service_account
@mock.patch('google.oauth2.id_token.verify_token', autospec=True)
def test_verify_firebase_token_clock_skew(verify_token):
    result = id_token.verify_firebase_token(mock.sentinel.token, mock.sentinel.request, audience=mock.sentinel.audience, clock_skew_in_seconds=10)
    assert result == verify_token.return_value
    verify_token.assert_called_once_with(mock.sentinel.token, mock.sentinel.request, audience=mock.sentinel.audience, certs_url=id_token._GOOGLE_APIS_CERTS_URL, clock_skew_in_seconds=10)