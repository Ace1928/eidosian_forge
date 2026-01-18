import datetime
import functools
import os
import sys
import freezegun
import mock
import OpenSSL
import pytest  # type: ignore
import requests
import requests.adapters
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._custom_tls_signer
import google.auth.transport._mtls_helper
import google.auth.transport.requests
from google.oauth2 import service_account
from tests.transport import compliance
def test_request_refresh(self):
    credentials = mock.Mock(wraps=CredentialsStub())
    final_response = make_response(status=http_client.OK)
    adapter = AdapterStub([make_response(status=http_client.UNAUTHORIZED), final_response])
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials, refresh_timeout=60)
    authed_session.mount(self.TEST_URL, adapter)
    result = authed_session.request('GET', self.TEST_URL)
    assert result == final_response
    assert credentials.before_request.call_count == 2
    assert credentials.refresh.called
    assert len(adapter.requests) == 2
    assert adapter.requests[0].url == self.TEST_URL
    assert adapter.requests[0].headers['authorization'] == 'token'
    assert adapter.requests[1].url == self.TEST_URL
    assert adapter.requests[1].headers['authorization'] == 'token1'