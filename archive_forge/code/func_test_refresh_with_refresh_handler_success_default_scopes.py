import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import credentials
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min)
def test_refresh_with_refresh_handler_success_default_scopes(self, unused_utcnow):
    expected_expiry = datetime.datetime.min + datetime.timedelta(seconds=2800)
    original_refresh_handler = mock.Mock(return_value=('UNUSED_TOKEN', expected_expiry))
    refresh_handler = mock.Mock(return_value=('ACCESS_TOKEN', expected_expiry))
    default_scopes = ['https://www.googleapis.com/auth/cloud-platform']
    request = mock.create_autospec(transport.Request)
    creds = credentials.Credentials(token=None, refresh_token=None, token_uri=None, client_id=None, client_secret=None, rapt_token=None, scopes=None, default_scopes=default_scopes, refresh_handler=original_refresh_handler)
    creds.refresh_handler = refresh_handler
    creds.refresh(request)
    assert creds.token == 'ACCESS_TOKEN'
    assert creds.expiry == expected_expiry
    assert creds.valid
    assert not creds.expired
    refresh_handler.assert_called_with(request, scopes=default_scopes)