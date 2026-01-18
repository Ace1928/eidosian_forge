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
def test_urlopen_no_default_host(self):
    credentials = mock.create_autospec(service_account.Credentials)
    authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials)
    authed_http.credentials._create_self_signed_jwt.assert_called_once_with(None)