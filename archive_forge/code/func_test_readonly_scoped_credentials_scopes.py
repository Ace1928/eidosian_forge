import datetime
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _helpers
def test_readonly_scoped_credentials_scopes():
    credentials = ReadOnlyScopedCredentialsImpl()
    credentials._scopes = ['one', 'two']
    assert credentials.scopes == ['one', 'two']
    assert credentials.has_scopes(['one'])
    assert credentials.has_scopes(['two'])
    assert credentials.has_scopes(['one', 'two'])
    assert not credentials.has_scopes(['three'])