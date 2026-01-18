import base64
import datetime
import mock
import pytest  # type: ignore
import responses  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.auth.compute_engine import credentials
from google.auth.transport import requests
def test_token_uri(self):
    request = mock.create_autospec(transport.Request, instance=True)
    self.credentials = credentials.IDTokenCredentials(request=request, signer=mock.Mock(), service_account_email='foo@example.com', target_audience='https://audience.com')
    assert self.credentials._token_uri == credentials._DEFAULT_TOKEN_URI
    self.credentials = credentials.IDTokenCredentials(request=request, signer=mock.Mock(), service_account_email='foo@example.com', target_audience='https://audience.com', token_uri='https://example.com/token')
    assert self.credentials._token_uri == 'https://example.com/token'