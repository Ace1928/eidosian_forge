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
class TestAuthorizedHttp(object):
    TEST_URL = 'http://example.com'

    def test_authed_http_defaults(self):
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(mock.sentinel.credentials)
        assert authed_http.credentials == mock.sentinel.credentials
        assert isinstance(authed_http.http, urllib3.PoolManager)

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

    def test_urlopen_refresh(self):
        credentials = mock.Mock(wraps=CredentialsStub())
        final_response = ResponseStub(status=http_client.OK)
        http = HttpStub([ResponseStub(status=http_client.UNAUTHORIZED), final_response])
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials, http=http)
        authed_http = authed_http.urlopen('GET', 'http://example.com')
        assert authed_http == final_response
        assert credentials.before_request.call_count == 2
        assert credentials.refresh.called
        assert http.requests == [('GET', self.TEST_URL, None, {'authorization': 'token'}, {}), ('GET', self.TEST_URL, None, {'authorization': 'token1'}, {})]

    def test_urlopen_no_default_host(self):
        credentials = mock.create_autospec(service_account.Credentials)
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials)
        authed_http.credentials._create_self_signed_jwt.assert_called_once_with(None)

    def test_urlopen_with_default_host(self):
        default_host = 'pubsub.googleapis.com'
        credentials = mock.create_autospec(service_account.Credentials)
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials, default_host=default_host)
        authed_http.credentials._create_self_signed_jwt.assert_called_once_with('https://{}/'.format(default_host))

    def test_proxies(self):
        http = mock.create_autospec(urllib3.PoolManager)
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(None, http=http)
        with authed_http:
            pass
        assert http.__enter__.called
        assert http.__exit__.called
        authed_http.headers = mock.sentinel.headers
        assert authed_http.headers == http.headers

    @mock.patch('google.auth.transport.urllib3._make_mutual_tls_http', autospec=True)
    def test_configure_mtls_channel_with_callback(self, mock_make_mutual_tls_http):
        callback = mock.Mock()
        callback.return_value = (pytest.public_cert_bytes, pytest.private_key_bytes)
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials=mock.Mock(), http=mock.Mock())
        with pytest.warns(UserWarning):
            with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
                is_mtls = authed_http.configure_mtls_channel(callback)
        assert is_mtls
        mock_make_mutual_tls_http.assert_called_once_with(cert=pytest.public_cert_bytes, key=pytest.private_key_bytes)

    @mock.patch('google.auth.transport.urllib3._make_mutual_tls_http', autospec=True)
    @mock.patch('google.auth.transport._mtls_helper.get_client_cert_and_key', autospec=True)
    def test_configure_mtls_channel_with_metadata(self, mock_get_client_cert_and_key, mock_make_mutual_tls_http):
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials=mock.Mock())
        mock_get_client_cert_and_key.return_value = (True, pytest.public_cert_bytes, pytest.private_key_bytes)
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            is_mtls = authed_http.configure_mtls_channel()
        assert is_mtls
        mock_get_client_cert_and_key.assert_called_once()
        mock_make_mutual_tls_http.assert_called_once_with(cert=pytest.public_cert_bytes, key=pytest.private_key_bytes)

    @mock.patch('google.auth.transport.urllib3._make_mutual_tls_http', autospec=True)
    @mock.patch('google.auth.transport._mtls_helper.get_client_cert_and_key', autospec=True)
    def test_configure_mtls_channel_non_mtls(self, mock_get_client_cert_and_key, mock_make_mutual_tls_http):
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials=mock.Mock())
        mock_get_client_cert_and_key.return_value = (False, None, None)
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            is_mtls = authed_http.configure_mtls_channel()
        assert not is_mtls
        mock_get_client_cert_and_key.assert_called_once()
        mock_make_mutual_tls_http.assert_not_called()

    @mock.patch('google.auth.transport._mtls_helper.get_client_cert_and_key', autospec=True)
    def test_configure_mtls_channel_exceptions(self, mock_get_client_cert_and_key):
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials=mock.Mock())
        mock_get_client_cert_and_key.side_effect = exceptions.ClientCertError()
        with pytest.raises(exceptions.MutualTLSChannelError):
            with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
                authed_http.configure_mtls_channel()
        mock_get_client_cert_and_key.return_value = (False, None, None)
        with mock.patch.dict('sys.modules'):
            sys.modules['OpenSSL'] = None
            with pytest.raises(exceptions.MutualTLSChannelError):
                with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
                    authed_http.configure_mtls_channel()

    @mock.patch('google.auth.transport._mtls_helper.get_client_cert_and_key', autospec=True)
    def test_configure_mtls_channel_without_client_cert_env(self, get_client_cert_and_key):
        callback = mock.Mock()
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials=mock.Mock(), http=mock.Mock())
        is_mtls = authed_http.configure_mtls_channel(callback)
        assert not is_mtls
        callback.assert_not_called()
        is_mtls = authed_http.configure_mtls_channel(callback)
        assert not is_mtls
        get_client_cert_and_key.assert_not_called()

    def test_clear_pool_on_del(self):
        http = mock.create_autospec(urllib3.PoolManager)
        authed_http = google.auth.transport.urllib3.AuthorizedHttp(mock.sentinel.credentials, http=http)
        authed_http.__del__()
        http.clear.assert_called_with()
        authed_http.http = None
        authed_http.__del__()