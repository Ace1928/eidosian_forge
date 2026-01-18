import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.oauth2 import _credentials_async as _credentials_async
from google.oauth2 import credentials
from tests.oauth2 import test_credentials
@mock.patch('google.auth._cloud_sdk.get_auth_access_token', autospec=True)
def test_refresh(self, get_auth_access_token):
    get_auth_access_token.return_value = 'access_token'
    cred = _credentials_async.UserAccessTokenCredentials()
    cred.refresh(None)
    assert cred.token == 'access_token'