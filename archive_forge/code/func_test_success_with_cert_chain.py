import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
@mock.patch('subprocess.Popen', autospec=True)
def test_success_with_cert_chain(self, mock_popen):
    PUBLIC_CERT_CHAIN_BYTES = pytest.public_cert_bytes + pytest.public_cert_bytes
    mock_popen.return_value = self.create_mock_process(PUBLIC_CERT_CHAIN_BYTES + pytest.private_key_bytes, b'')
    cert, key, passphrase = _mtls_helper._run_cert_provider_command(['command'])
    assert cert == PUBLIC_CERT_CHAIN_BYTES
    assert key == pytest.private_key_bytes
    assert passphrase is None
    mock_popen.return_value = self.create_mock_process(PUBLIC_CERT_CHAIN_BYTES + ENCRYPTED_EC_PRIVATE_KEY + PASSPHRASE, b'')
    cert, key, passphrase = _mtls_helper._run_cert_provider_command(['command'], expect_encrypted_key=True)
    assert cert == PUBLIC_CERT_CHAIN_BYTES
    assert key == ENCRYPTED_EC_PRIVATE_KEY
    assert passphrase == PASSPHRASE_VALUE