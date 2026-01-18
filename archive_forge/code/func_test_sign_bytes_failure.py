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
def test_sign_bytes_failure(self):
    credentials = self.make_credentials(lifetime=None)
    with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
        data = {'error': {'code': 403, 'message': 'unauthorized'}}
        auth_session.return_value = MockResponse(data, http_client.FORBIDDEN)
        with pytest.raises(exceptions.TransportError) as excinfo:
            credentials.sign_bytes(b'foo')
        assert excinfo.match("'code': 403")