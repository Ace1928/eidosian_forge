import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
@mock.patch('subprocess.Popen', autospec=True)
def test_passphrase_not_expected(self, mock_popen):
    mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes + pytest.private_key_bytes + PASSPHRASE, b'')
    with pytest.raises(exceptions.ClientCertError):
        _mtls_helper._run_cert_provider_command(['command'])