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
class TestMutualTlsAdapter(object):

    @mock.patch.object(requests.adapters.HTTPAdapter, 'init_poolmanager')
    @mock.patch.object(requests.adapters.HTTPAdapter, 'proxy_manager_for')
    def test_success(self, mock_proxy_manager_for, mock_init_poolmanager):
        adapter = google.auth.transport.requests._MutualTlsAdapter(pytest.public_cert_bytes, pytest.private_key_bytes)
        adapter.init_poolmanager()
        mock_init_poolmanager.assert_called_with(ssl_context=adapter._ctx_poolmanager)
        adapter.proxy_manager_for()
        mock_proxy_manager_for.assert_called_with(ssl_context=adapter._ctx_proxymanager)

    def test_invalid_cert_or_key(self):
        with pytest.raises(OpenSSL.crypto.Error):
            google.auth.transport.requests._MutualTlsAdapter(b'invalid cert', b'invalid key')

    @mock.patch.dict('sys.modules', {'OpenSSL.crypto': None})
    def test_import_error(self):
        with pytest.raises(ImportError):
            google.auth.transport.requests._MutualTlsAdapter(pytest.public_cert_bytes, pytest.private_key_bytes)