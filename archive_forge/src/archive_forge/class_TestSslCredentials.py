import datetime
import os
import time
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import credentials
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account
@mock.patch('grpc.ssl_channel_credentials', autospec=True)
@mock.patch('google.auth.transport._mtls_helper.get_client_ssl_credentials', autospec=True)
@mock.patch('google.auth.transport._mtls_helper._read_dca_metadata_file', autospec=True)
@mock.patch('google.auth.transport._mtls_helper._check_dca_metadata_path', autospec=True)
class TestSslCredentials(object):

    def test_no_context_aware_metadata(self, mock_check_dca_metadata_path, mock_read_dca_metadata_file, mock_get_client_ssl_credentials, mock_ssl_channel_credentials):
        mock_check_dca_metadata_path.return_value = None
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            ssl_credentials = google.auth.transport.grpc.SslCredentials()
        assert ssl_credentials.ssl_credentials is not None
        assert not ssl_credentials.is_mtls
        mock_get_client_ssl_credentials.assert_not_called()
        mock_ssl_channel_credentials.assert_called_once_with()

    def test_get_client_ssl_credentials_failure(self, mock_check_dca_metadata_path, mock_read_dca_metadata_file, mock_get_client_ssl_credentials, mock_ssl_channel_credentials):
        mock_check_dca_metadata_path.return_value = METADATA_PATH
        mock_read_dca_metadata_file.return_value = {'cert_provider_command': ['some command']}
        mock_get_client_ssl_credentials.side_effect = exceptions.ClientCertError()
        with pytest.raises(exceptions.MutualTLSChannelError):
            with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
                assert google.auth.transport.grpc.SslCredentials().ssl_credentials

    def test_get_client_ssl_credentials_success(self, mock_check_dca_metadata_path, mock_read_dca_metadata_file, mock_get_client_ssl_credentials, mock_ssl_channel_credentials):
        mock_check_dca_metadata_path.return_value = METADATA_PATH
        mock_read_dca_metadata_file.return_value = {'cert_provider_command': ['some command']}
        mock_get_client_ssl_credentials.return_value = (True, PUBLIC_CERT_BYTES, PRIVATE_KEY_BYTES, None)
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            ssl_credentials = google.auth.transport.grpc.SslCredentials()
        assert ssl_credentials.ssl_credentials is not None
        assert ssl_credentials.is_mtls
        mock_get_client_ssl_credentials.assert_called_once()
        mock_ssl_channel_credentials.assert_called_once_with(certificate_chain=PUBLIC_CERT_BYTES, private_key=PRIVATE_KEY_BYTES)

    def test_get_client_ssl_credentials_without_client_cert_env(self, mock_check_dca_metadata_path, mock_read_dca_metadata_file, mock_get_client_ssl_credentials, mock_ssl_channel_credentials):
        ssl_credentials = google.auth.transport.grpc.SslCredentials()
        assert ssl_credentials.ssl_credentials is not None
        assert not ssl_credentials.is_mtls
        mock_check_dca_metadata_path.assert_not_called()
        mock_read_dca_metadata_file.assert_not_called()
        mock_get_client_ssl_credentials.assert_not_called()
        mock_ssl_channel_credentials.assert_called_once()