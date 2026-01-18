import pytest
from google.auth import _helpers
from google.auth import exceptions
from google.auth import iam
from google.oauth2 import service_account
def test_refresh_success(http_request, credentials, token_info):
    credentials = credentials.with_scopes(['email', 'profile'])
    credentials.refresh(http_request)
    assert credentials.token
    info = token_info(credentials.token)
    assert info['email'] == credentials.service_account_email
    info_scopes = _helpers.string_to_scopes(info['scope'])
    assert set(info_scopes) == set(['https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'])