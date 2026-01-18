import datetime
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _service_account_async as service_account
from tests.oauth2 import test_service_account
def test_sign_bytes(self):
    credentials = self.make_credentials()
    to_sign = b'123'
    signature = credentials.sign_bytes(to_sign)
    assert crypt.verify_signature(to_sign, signature, test_service_account.PUBLIC_CERT_BYTES)