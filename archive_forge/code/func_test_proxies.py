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
def test_proxies(self):
    http = mock.create_autospec(urllib3.PoolManager)
    authed_http = google.auth.transport.urllib3.AuthorizedHttp(None, http=http)
    with authed_http:
        pass
    assert http.__enter__.called
    assert http.__exit__.called
    authed_http.headers = mock.sentinel.headers
    assert authed_http.headers == http.headers