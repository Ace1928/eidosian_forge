import base64
import json
import os
from cryptography.hazmat.primitives.asymmetric import ec
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import base
from google.auth.crypt import es256
class TestES256Verifier(object):

    def test_verify_success(self):
        to_sign = b'foo'
        signer = es256.ES256Signer.from_string(PRIVATE_KEY_BYTES)
        actual_signature = signer.sign(to_sign)
        verifier = es256.ES256Verifier.from_string(PUBLIC_KEY_BYTES)
        assert verifier.verify(to_sign, actual_signature)

    def test_verify_unicode_success(self):
        to_sign = u'foo'
        signer = es256.ES256Signer.from_string(PRIVATE_KEY_BYTES)
        actual_signature = signer.sign(to_sign)
        verifier = es256.ES256Verifier.from_string(PUBLIC_KEY_BYTES)
        assert verifier.verify(to_sign, actual_signature)

    def test_verify_failure(self):
        verifier = es256.ES256Verifier.from_string(PUBLIC_KEY_BYTES)
        bad_signature1 = b''
        assert not verifier.verify(b'foo', bad_signature1)
        bad_signature2 = b'a'
        assert not verifier.verify(b'foo', bad_signature2)

    def test_verify_failure_with_wrong_raw_signature(self):
        to_sign = b'foo'
        wrong_signature = base64.urlsafe_b64decode(b'm7oaRxUDeYqjZ8qiMwo0PZLTMZWKJLFQREpqce1StMIa_yXQQ-C5WgeIRHW7OqlYSDL0XbUrj_uAw9i-QhfOJQ==')
        verifier = es256.ES256Verifier.from_string(PUBLIC_KEY_BYTES)
        assert not verifier.verify(to_sign, wrong_signature)

    def test_from_string_pub_key(self):
        verifier = es256.ES256Verifier.from_string(PUBLIC_KEY_BYTES)
        assert isinstance(verifier, es256.ES256Verifier)
        assert isinstance(verifier._pubkey, ec.EllipticCurvePublicKey)

    def test_from_string_pub_key_unicode(self):
        public_key = _helpers.from_bytes(PUBLIC_KEY_BYTES)
        verifier = es256.ES256Verifier.from_string(public_key)
        assert isinstance(verifier, es256.ES256Verifier)
        assert isinstance(verifier._pubkey, ec.EllipticCurvePublicKey)

    def test_from_string_pub_cert(self):
        verifier = es256.ES256Verifier.from_string(PUBLIC_CERT_BYTES)
        assert isinstance(verifier, es256.ES256Verifier)
        assert isinstance(verifier._pubkey, ec.EllipticCurvePublicKey)

    def test_from_string_pub_cert_unicode(self):
        public_cert = _helpers.from_bytes(PUBLIC_CERT_BYTES)
        verifier = es256.ES256Verifier.from_string(public_cert)
        assert isinstance(verifier, es256.ES256Verifier)
        assert isinstance(verifier._pubkey, ec.EllipticCurvePublicKey)