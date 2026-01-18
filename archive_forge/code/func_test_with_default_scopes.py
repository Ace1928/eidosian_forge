import datetime
import mock
import pytest  # type: ignore
from google.auth import app_engine
def test_with_default_scopes(self, app_identity):
    credentials = app_engine.Credentials()
    assert not credentials.scopes
    assert not credentials.default_scopes
    assert credentials.requires_scopes
    scoped_credentials = credentials.with_scopes(scopes=None, default_scopes=['email'])
    assert scoped_credentials.has_scopes(['email'])
    assert not scoped_credentials.requires_scopes