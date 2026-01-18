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
def test_with_scopes_using_user_and_default_scopes(self):
    credentials = self.make_credentials()
    assert not credentials.scopes
    assert credentials.requires_scopes
    scoped_credentials = credentials.with_scopes(['email'], default_scopes=['profile'])
    assert scoped_credentials.has_scopes(['email'])
    assert not scoped_credentials.has_scopes(['profile'])
    assert not scoped_credentials.requires_scopes
    assert scoped_credentials.scopes == ['email']
    assert scoped_credentials.default_scopes == ['profile']