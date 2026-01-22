import json
import os
from cryptography.hazmat.primitives.asymmetric import rsa
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import _cryptography_rsa
from google.auth.crypt import base
class TestRSAVerifier(object):

    def test_verify_success(self):
        to_sign = b'foo'
        signer = _cryptography_rsa.RSASigner.from_string(PRIVATE_KEY_BYTES)
        actual_signature = signer.sign(to_sign)
        verifier = _cryptography_rsa.RSAVerifier.from_string(PUBLIC_KEY_BYTES)
        assert verifier.verify(to_sign, actual_signature)

    def test_verify_unicode_success(self):
        to_sign = u'foo'
        signer = _cryptography_rsa.RSASigner.from_string(PRIVATE_KEY_BYTES)
        actual_signature = signer.sign(to_sign)
        verifier = _cryptography_rsa.RSAVerifier.from_string(PUBLIC_KEY_BYTES)
        assert verifier.verify(to_sign, actual_signature)

    def test_verify_failure(self):
        verifier = _cryptography_rsa.RSAVerifier.from_string(PUBLIC_KEY_BYTES)
        bad_signature1 = b''
        assert not verifier.verify(b'foo', bad_signature1)
        bad_signature2 = b'a'
        assert not verifier.verify(b'foo', bad_signature2)

    def test_from_string_pub_key(self):
        verifier = _cryptography_rsa.RSAVerifier.from_string(PUBLIC_KEY_BYTES)
        assert isinstance(verifier, _cryptography_rsa.RSAVerifier)
        assert isinstance(verifier._pubkey, rsa.RSAPublicKey)

    def test_from_string_pub_key_unicode(self):
        public_key = _helpers.from_bytes(PUBLIC_KEY_BYTES)
        verifier = _cryptography_rsa.RSAVerifier.from_string(public_key)
        assert isinstance(verifier, _cryptography_rsa.RSAVerifier)
        assert isinstance(verifier._pubkey, rsa.RSAPublicKey)

    def test_from_string_pub_cert(self):
        verifier = _cryptography_rsa.RSAVerifier.from_string(PUBLIC_CERT_BYTES)
        assert isinstance(verifier, _cryptography_rsa.RSAVerifier)
        assert isinstance(verifier._pubkey, rsa.RSAPublicKey)

    def test_from_string_pub_cert_unicode(self):
        public_cert = _helpers.from_bytes(PUBLIC_CERT_BYTES)
        verifier = _cryptography_rsa.RSAVerifier.from_string(public_cert)
        assert isinstance(verifier, _cryptography_rsa.RSAVerifier)
        assert isinstance(verifier._pubkey, rsa.RSAPublicKey)