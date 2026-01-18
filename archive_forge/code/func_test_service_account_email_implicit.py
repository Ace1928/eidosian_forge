import datetime
import mock
import pytest  # type: ignore
from google.auth import app_engine
def test_service_account_email_implicit(self, app_identity):
    app_identity.get_service_account_name.return_value = mock.sentinel.service_account_email
    credentials = app_engine.Credentials()
    assert credentials.service_account_email == mock.sentinel.service_account_email
    assert app_identity.get_service_account_name.called