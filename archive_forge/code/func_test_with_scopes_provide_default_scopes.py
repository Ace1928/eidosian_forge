import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import transport
from google.auth.impersonated_credentials import Credentials
from google.oauth2 import credentials
from google.oauth2 import service_account
def test_with_scopes_provide_default_scopes(self):
    credentials = self.make_credentials()
    credentials._target_scopes = []
    credentials = credentials.with_scopes(['fake_scope1'], default_scopes=['fake_scope2'])
    assert credentials._target_scopes == ['fake_scope1']