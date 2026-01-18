import os
import sys
import mock
import OpenSSL
import pytest  # type: ignore
from six.moves import http_client
import urllib3  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._mtls_helper
import google.auth.transport.urllib3
from google.oauth2 import service_account
from tests.transport import compliance
def test_urlopen_no_refresh(self):
    credentials = mock.Mock(wraps=CredentialsStub())
    response = ResponseStub()
    http = HttpStub([response])
    authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials, http=http)
    result = authed_http.urlopen('GET', self.TEST_URL)
    assert result == response
    assert credentials.before_request.called
    assert not credentials.refresh.called
    assert http.requests == [('GET', self.TEST_URL, None, {'authorization': 'token'}, {})]