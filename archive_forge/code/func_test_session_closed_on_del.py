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
def test_session_closed_on_del(self):
    http = mock.create_autospec(requests.Session, instance=True)
    request = google.auth.transport.requests.Request(http)
    request.__del__()
    http.close.assert_called_with()
    http = mock.create_autospec(requests.Session, instance=True)
    http.close.side_effect = TypeError('test injected TypeError')
    request = google.auth.transport.requests.Request(http)
    request.__del__()
    http.close.assert_called_with()