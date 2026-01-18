import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
@mock.patch('google.auth.transport._mtls_helper._run_cert_provider_command', autospec=True)
@mock.patch('google.auth.transport._mtls_helper._read_dca_metadata_file', autospec=True)
@mock.patch('google.auth.transport._mtls_helper._check_dca_metadata_path', autospec=True)
def test_success_with_encrypted_key(self, mock_check_dca_metadata_path, mock_read_dca_metadata_file, mock_run_cert_provider_command):
    mock_check_dca_metadata_path.return_value = True
    mock_read_dca_metadata_file.return_value = {'cert_provider_command': ['command']}
    mock_run_cert_provider_command.return_value = (b'cert', b'key', b'passphrase')
    has_cert, cert, key, passphrase = _mtls_helper.get_client_ssl_credentials(generate_encrypted_key=True)
    assert has_cert
    assert cert == b'cert'
    assert key == b'key'
    assert passphrase == b'passphrase'
    mock_run_cert_provider_command.assert_called_once_with(['command', '--with_passphrase'], expect_encrypted_key=True)