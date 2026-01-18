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
@responses.activate
def test_with_target_audience_integration(self):
    """ Test that it is possible to refresh credentials
        generated from `with_target_audience`.

        Instead of mocking the methods, the HTTP responses
        have been mocked.
        """
    responses.add(responses.GET, 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true', status=200, content_type='application/json', json={'scopes': 'email', 'email': 'service-account@example.com', 'aliases': ['default']})
    responses.add(responses.GET, 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/service-account@example.com/token', status=200, content_type='application/json', json={'access_token': 'some-token', 'expires_in': 3210, 'token_type': 'Bearer'})
    signature = base64.b64encode(b'some-signature').decode('utf-8')
    responses.add(responses.POST, 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/service-account@example.com:signBlob?alt=json', status=200, content_type='application/json', json={'keyId': 'some-key-id', 'signedBlob': signature})
    id_token = '{}.{}.{}'.format(base64.b64encode(b'{"some":"some"}').decode('utf-8'), base64.b64encode(b'{"exp": 3210}').decode('utf-8'), base64.b64encode(b'token').decode('utf-8'))
    responses.add(responses.POST, 'https://www.googleapis.com/oauth2/v4/token', status=200, content_type='application/json', json={'id_token': id_token, 'expiry': 3210})
    self.credentials = credentials.IDTokenCredentials(request=requests.Request(), service_account_email='service-account@example.com', target_audience='https://audience.com')
    self.credentials = self.credentials.with_target_audience('https://actually.not')
    self.credentials.refresh(requests.Request())
    assert self.credentials.token is not None