import datetime
import mock
import pytest  # type: ignore
from google.auth import app_engine
class TestSigner(object):

    def test_key_id(self, app_identity):
        app_identity.sign_blob.return_value = (mock.sentinel.key_id, mock.sentinel.signature)
        signer = app_engine.Signer()
        assert signer.key_id is None

    def test_sign(self, app_identity):
        app_identity.sign_blob.return_value = (mock.sentinel.key_id, mock.sentinel.signature)
        signer = app_engine.Signer()
        to_sign = b'123'
        signature = signer.sign(to_sign)
        assert signature == mock.sentinel.signature
        app_identity.sign_blob.assert_called_with(to_sign)