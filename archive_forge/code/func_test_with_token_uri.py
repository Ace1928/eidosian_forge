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
def test_with_token_uri(self):
    info = AUTH_USER_INFO.copy()
    creds = credentials.Credentials.from_authorized_user_info(info)
    new_token_uri = 'https://oauth2-eu.googleapis.com/token'
    assert creds._token_uri == credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT
    creds_with_new_token_uri = creds.with_token_uri(new_token_uri)
    assert creds_with_new_token_uri._token_uri == new_token_uri